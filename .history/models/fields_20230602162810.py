import torch
import torch.nn as nn
import numpy as np
from models.embedder import get_embedder
import tinycudann as tcnn
from models.NeRF import MLP, MLP_v1
from functools import partial


class Shift(nn.Module):
    def __init__(self, shift) -> None:
        super().__init__()
        self.shift = shift


    def forward(self, x):
        return x + self.shift


class BaseProjectionLayer(nn.Module):
    @property
    def proj_dims(self):
        raise NotImplementedError()


    def forward(self, x):
        raise NotImplementedError()


class ProjectionLayer(BaseProjectionLayer):
    def __init__(self, input_dims, proj_dims):
        super().__init__()
        self._proj_dims = proj_dims

        self.proj = nn.Sequential(
            nn.Linear(input_dims, 2 * proj_dims), nn.ReLU(), nn.Linear(2 * proj_dims, proj_dims)
        )


    @property
    def proj_dims(self):
        return self._proj_dims


    def forward(self, x):
        return self.proj(x)


class CouplingLayer(nn.Module):
    def __init__(self, map_s, map_t, projection, mask):
        super().__init__()
        self.map_s = map_s
        self.map_t = map_t
        self.projection = projection
        self.register_buffer("mask", mask) # 1,1,1,3 -> 1,3


    def forward(self, F, y, alpha_ratio):
        y1 = y * self.mask

        F_y1 = torch.cat([F, self.projection(y1,alpha_ratio)], dim=-1)
        s = self.map_s(F_y1)
        t = self.map_t(F_y1)

        x = y1 + (1 - self.mask) * ((y - t) * torch.exp(-s))
        ldj = (-s).sum(-1)

        return x, ldj


    def inverse(self, F, x, alpha_ratio):
        x1 = x * self.mask

        F_x1 = torch.cat([F, self.projection(x1,alpha_ratio)], dim=-1)
        s = self.map_s(F_x1)
        t = self.map_t(F_x1)

        y = x1 + (1 - self.mask) * (x * torch.exp(s) + t)
        ldj = s.sum(-1)

        return y, ldj


def euler2rot(euler_angle):
    batch_size = euler_angle.shape[0]
    one = torch.ones(batch_size, 1, 1).to(euler_angle.device)
    zero = torch.zeros(batch_size, 1, 1).to(euler_angle.device)
    theta = euler_angle[:, 0].reshape(-1, 1, 1)
    phi = euler_angle[:, 1].reshape(-1, 1, 1)
    psi = euler_angle[:, 2].reshape(-1, 1, 1)
    rot_x = torch.cat((
        torch.cat((one, zero, zero), 1),
        torch.cat((zero, theta.cos(), theta.sin()), 1),
        torch.cat((zero, -theta.sin(), theta.cos()), 1),
    ), 2)
    rot_y = torch.cat((
        torch.cat((phi.cos(), zero, -phi.sin()), 1),
        torch.cat((zero, one, zero), 1),
        torch.cat((phi.sin(), zero, phi.cos()), 1),
    ), 2)
    rot_z = torch.cat((
        torch.cat((psi.cos(), -psi.sin(), zero), 1),
        torch.cat((psi.sin(), psi.cos(), zero), 1),
        torch.cat((zero, zero, one), 1)
    ), 2)
    return torch.bmm(rot_x, torch.bmm(rot_y, rot_z))


def euler2rot_inv(euler_angle):
    batch_size = euler_angle.shape[0]
    one = torch.ones(batch_size, 1, 1).to(euler_angle.device)
    zero = torch.zeros(batch_size, 1, 1).to(euler_angle.device)
    theta = euler_angle[:, 0].reshape(-1, 1, 1)
    phi = euler_angle[:, 1].reshape(-1, 1, 1)
    psi = euler_angle[:, 2].reshape(-1, 1, 1)
    rot_x = torch.cat((
        torch.cat((one, zero, zero), 1),
        torch.cat((zero, theta.cos(), theta.sin()), 1),
        torch.cat((zero, -theta.sin(), theta.cos()), 1),
    ), 2)
    rot_y = torch.cat((
        torch.cat((phi.cos(), zero, -phi.sin()), 1),
        torch.cat((zero, one, zero), 1),
        torch.cat((phi.sin(), zero, phi.cos()), 1),
    ), 2)
    rot_z = torch.cat((
        torch.cat((psi.cos(), -psi.sin(), zero), 1),
        torch.cat((psi.sin(), psi.cos(), zero), 1),
        torch.cat((zero, zero, one), 1)
    ), 2)
    return torch.bmm(rot_z, torch.bmm(rot_y, rot_x))


def euler2rot_2d(euler_angle):
    # (B, 1) -> (B, 2, 2)
    theta = euler_angle.reshape(-1, 1, 1)
    rot = torch.cat((
        torch.cat((theta.cos(), theta.sin()), 1),
        torch.cat((-theta.sin(), theta.cos()), 1),
    ), 2)

    return rot


def euler2rot_2dinv(euler_angle):
    # (B, 1) -> (B, 2, 2)
    theta = euler_angle.reshape(-1, 1, 1)
    rot = torch.cat((
        torch.cat((theta.cos(), -theta.sin()), 1),
        torch.cat((theta.sin(), theta.cos()), 1),
    ), 2)

    return rot


def quaternions_to_rotation_matrices(quaternions):
    """
    Arguments:
    ---------
        quaternions: Tensor with size ...x4, where ... denotes any shape of
                     quaternions to be translated to rotation matrices
    Returns:
    -------
        rotation_matrices: Tensor with size ...x3x3, that contains the computed
                           rotation matrices
    """
    # Allocate memory for a Tensor of size ...x3x3 that will hold the rotation
    # matrix along the x-axis
    shape = quaternions.shape[:-1] + (3, 3)
    R = quaternions.new_zeros(shape)

    # A unit quaternion is q = w + xi + yj + zk
    xx = quaternions[..., 1] ** 2
    yy = quaternions[..., 2] ** 2
    zz = quaternions[..., 3] ** 2
    ww = quaternions[..., 0] ** 2
    n = (ww + xx + yy + zz).unsqueeze(-1)
    s = torch.zeros_like(n)
    s[n != 0] = 2 / n[n != 0]

    xy = s[..., 0] * quaternions[..., 1] * quaternions[..., 2]
    xz = s[..., 0] * quaternions[..., 1] * quaternions[..., 3]
    yz = s[..., 0] * quaternions[..., 2] * quaternions[..., 3]
    xw = s[..., 0] * quaternions[..., 1] * quaternions[..., 0]
    yw = s[..., 0] * quaternions[..., 2] * quaternions[..., 0]
    zw = s[..., 0] * quaternions[..., 3] * quaternions[..., 0]

    xx = s[..., 0] * xx
    yy = s[..., 0] * yy
    zz = s[..., 0] * zz

    R[..., 0, 0] = 1 - yy - zz
    R[..., 0, 1] = xy - zw
    R[..., 0, 2] = xz + yw

    R[..., 1, 0] = xy + zw
    R[..., 1, 1] = 1 - xx - zz
    R[..., 1, 2] = yz - xw

    R[..., 2, 0] = xz - yw
    R[..., 2, 1] = yz + xw
    R[..., 2, 2] = 1 - xx - yy

    return R


# Deform

class DeformNetwork_MLP(nn.Module):
    def __init__(self,
                 d_feature,
                 d_in,
                 d_out_1,
                 d_out_2,
                 d_hidden,
                 n_layers,
                 skip_in=(4,),
                 multires=0,
                 weight_norm=True,
                 normalizing_flow = False):
        super(DeformNetwork_MLP, self).__init__()
        
        self.skip_in = skip_in

        # part a
        # xy -> z
        ori_in = d_in
        dims_in = ori_in
        dims = [dims_in + d_feature] + [d_hidden for _ in range(n_layers)] + [d_out_1]

        self.embed_fn_1 = None

        if multires > 0:
            embed_fn, input_ch, warp_em = get_embedder(multires, use_alpha = True, input_dims=dims_in)
            self.embed_fn_1 = embed_fn
            dims_in = input_ch
            dims[0] = input_ch + d_feature
            self.warp_em = warp_em
        

        # latent code
        self.deform = MLP(in_ch = dims[0], out_ch=3, depth=n_layers, width=d_hidden,  hidden_activation=torch.nn.ReLU(), 
                    hidden_init= nn.init.xavier_normal_,
                    output_init=partial(nn.init.uniform_,b=1e-4))



    def forward(self, deformation_code, input_pts, alpha_ratio):
        batch_size = input_pts.shape[0]
        x = input_pts
        x = self.embed_fn_1(x)
        # x = torch.cat([deformation_code.repeat(x.shape[0],1), x], 1)
        x = torch.cat([deformation_code, x], 1)
        translate = self.deform(x)
        return input_pts + translate

    def inverse(self, deformation_code, input_pts, alpha_ratio):
        return torch.zeros_like(input_pts).to(input_pts)

class DeformNetwork(nn.Module):
    def __init__(self,
                 d_feature,
                 d_in,
                 d_out_1,
                 d_out_2,
                 n_blocks,
                 d_hidden,
                 n_layers,
                 skip_in=(4,),
                 multires=0,
                 weight_norm=True,
                 normalizing_flow = True):
        super(DeformNetwork, self).__init__()
        
        self.n_blocks = n_blocks
        self.skip_in = skip_in

        # part a
        # xy -> z
        ori_in = d_in - 1
        dims_in = ori_in
        dims = [dims_in + d_feature] + [d_hidden for _ in range(n_layers)] + [d_out_1]

        self.embed_fn_1 = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=dims_in)
            self.embed_fn_1 = embed_fn
            dims_in = input_ch
            dims[0] = input_ch + d_feature

        self.num_layers_a = len(dims)
        for i_b in range(self.n_blocks):
            for l in range(0, self.num_layers_a - 1):
                if l + 1 in self.skip_in:
                    out_dim = dims[l + 1] - dims_in
                else:
                    out_dim = dims[l + 1]

                lin = nn.Linear(dims[l], out_dim)

                if l == self.num_layers_a - 2:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight, 0.0)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight[:, :ori_in], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, ori_in:], 0.0)
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight[:, :-(dims_in - ori_in)], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims_in - ori_in):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

                if weight_norm and l < self.num_layers_a - 2:
                    lin = nn.utils.weight_norm(lin)

                setattr(self, "lin"+str(i_b)+"_a_"+str(l), lin)

        # part b
        # z -> xy
        ori_in = 1
        dims_in = ori_in
        dims = [dims_in + d_feature] + [d_hidden] + [d_out_2]

        self.embed_fn_2 = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=dims_in)
            self.embed_fn_2 = embed_fn
            dims_in = input_ch
            dims[0] = input_ch + d_feature

        self.num_layers_b = len(dims)
        for i_b in range(self.n_blocks):
            for l in range(0, self.num_layers_b - 1):
                if l + 1 in self.skip_in:
                    out_dim = dims[l + 1] - dims_in
                else:
                    out_dim = dims[l + 1]

                lin = nn.Linear(dims[l], out_dim)

                if l == self.num_layers_b - 2:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight, 0.0)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight[:, :ori_in], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, ori_in:], 0.0)
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight[:, :-(dims_in - ori_in)], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims_in - ori_in):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

                if weight_norm and l < self.num_layers_b - 2:
                    lin = nn.utils.weight_norm(lin)

                setattr(self, "lin"+str(i_b)+"_b_"+str(l), lin)

        # latent code
        for i_b in range(self.n_blocks):
            lin = nn.Linear(d_feature, d_feature)
            torch.nn.init.constant_(lin.bias, 0.0)
            torch.nn.init.constant_(lin.weight, 0.0)
            setattr(self, "lin"+str(i_b)+"_c", lin)

        self.activation = nn.Softplus(beta=100)


    def forward(self, deformation_code, input_pts, alpha_ratio):
        batch_size = input_pts.shape[0]
        x = input_pts
        for i_b in range(self.n_blocks):
            form = (i_b // 3) % 2
            mode = i_b % 3

            lin = getattr(self, "lin"+str(i_b)+"_c")
            deform_code_ib = lin(deformation_code) + deformation_code
            # deform_code_ib = deform_code_ib.repeat(batch_size, 1)
            # part a
            if form == 0:
                # zyx
                if mode == 0:
                    x_focus = x[:, [2]]
                    x_other = x[:, [0,1]]
                elif mode == 1:
                    x_focus = x[:, [1]]
                    x_other = x[:, [0,2]]
                else:
                    x_focus = x[:, [0]]
                    x_other = x[:, [1,2]]
            else:
                # xyz
                if mode == 0:
                    x_focus = x[:, [0]]
                    x_other = x[:, [1,2]]
                elif mode == 1:
                    x_focus = x[:, [1]]
                    x_other = x[:, [0,2]]
                else:
                    x_focus = x[:, [2]]
                    x_other = x[:, [0,1]]
            x_ori = x_other # xy
            if self.embed_fn_1 is not None:
                # Anneal
                x_other = self.embed_fn_1(x_other, alpha_ratio)
            x_other = torch.cat([x_other, deform_code_ib], dim=-1)
            x = x_other
            for l in range(0, self.num_layers_a - 1):
                lin = getattr(self, "lin"+str(i_b)+"_a_"+str(l))
                if l in self.skip_in:
                    x = torch.cat([x, x_other], 1) / np.sqrt(2)
                x = lin(x)
                if l < self.num_layers_a - 2:
                    x = self.activation(x)

            x_focus = x_focus - x

            # part b
            x_focus_ori = x_focus # z'
            if self.embed_fn_2 is not None:
                # Anneal
                x_focus = self.embed_fn_2(x_focus, alpha_ratio)
            x_focus = torch.cat([x_focus, deform_code_ib], dim=-1)
            x = x_focus
            for l in range(0, self.num_layers_b - 1):
                lin = getattr(self, "lin"+str(i_b)+"_b_"+str(l))
                if l in self.skip_in:
                    x = torch.cat([x, x_focus], 1) / np.sqrt(2)
                x = lin(x)
                if l < self.num_layers_b - 2:
                    x = self.activation(x)

            rot_2d = euler2rot_2dinv(x[:, [0]])
            trans_2d = x[:, 1:]
            x_other = torch.bmm(rot_2d, (x_ori - trans_2d)[...,None]).squeeze(-1)
            if form == 0:
                if mode == 0:
                    x = torch.cat([x_other, x_focus_ori], dim=-1)
                elif mode == 1:
                    x = torch.cat([x_other[:,[0]], x_focus_ori, x_other[:,[1]]], dim=-1)
                else:
                    x = torch.cat([x_focus_ori, x_other], dim=-1)
            else:
                if mode == 0:
                    x = torch.cat([x_focus_ori, x_other], dim=-1)
                elif mode == 1:
                    x = torch.cat([x_other[:,[0]], x_focus_ori, x_other[:,[1]]], dim=-1)
                else:
                    x = torch.cat([x_other, x_focus_ori], dim=-1)

        return x


    def inverse(self, deformation_code, input_pts, alpha_ratio):
        batch_size = input_pts.shape[0]
        x = input_pts
        for i_b in range(self.n_blocks):
            i_b = self.n_blocks - 1 - i_b # inverse
            form = (i_b // 3) % 2
            mode = i_b % 3

            lin = getattr(self, "lin"+str(i_b)+"_c")
            deform_code_ib = lin(deformation_code) + deformation_code
            deform_code_ib = deform_code_ib.repeat(batch_size, 1)
            # part b
            if form == 0:
                # axis: z -> y -> x
                if mode == 0:
                    x_focus = x[:, [0,1]]
                    x_other = x[:, [2]]
                elif mode == 1:
                    x_focus = x[:, [0,2]]
                    x_other = x[:, [1]]
                else:
                    x_focus = x[:, [1,2]]
                    x_other = x[:, [0]]
            else:
                # axis: x -> y -> z
                if mode == 0:
                    x_focus = x[:, [1,2]]
                    x_other = x[:, [0]]
                elif mode == 1:
                    x_focus = x[:, [0,2]]
                    x_other = x[:, [1]]
                else:
                    x_focus = x[:, [0,1]]
                    x_other = x[:, [2]]
            x_ori = x_other # z'
            if self.embed_fn_2 is not None:
                # Anneal
                x_other = self.embed_fn_2(x_other, alpha_ratio)
            x_other = torch.cat([x_other, deform_code_ib], dim=-1)
            x = x_other
            for l in range(0, self.num_layers_b - 1):
                lin = getattr(self, "lin"+str(i_b)+"_b_"+str(l))
                if l in self.skip_in:
                    x = torch.cat([x, x_other], 1) / np.sqrt(2)
                x = lin(x)
                if l < self.num_layers_b - 2:
                    x = self.activation(x)

            rot_2d = euler2rot_2d(x[:, [0]])
            trans_2d = x[:, 1:]
            x_focus = torch.bmm(rot_2d, x_focus[...,None]).squeeze(-1) + trans_2d

            # part a
            x_focus_ori = x_focus # xy
            if self.embed_fn_1 is not None:
                # Anneal
                x_focus = self.embed_fn_1(x_focus, alpha_ratio)
            x_focus = torch.cat([x_focus, deform_code_ib], dim=-1)
            x = x_focus
            for l in range(0, self.num_layers_a - 1):
                lin = getattr(self, "lin"+str(i_b)+"_a_"+str(l))
                if l in self.skip_in:
                    x = torch.cat([x, x_focus], 1) / np.sqrt(2)
                x = lin(x)
                if l < self.num_layers_a - 2:
                    x = self.activation(x)

            x_other = x_ori + x
            if form == 0:
                if mode == 0:
                    x = torch.cat([x_focus_ori, x_other], dim=-1)
                elif mode == 1:
                    x = torch.cat([x_focus_ori[:,[0]], x_other, x_focus_ori[:,[1]]], dim=-1)
                else:
                    x = torch.cat([x_other, x_focus_ori], dim=-1)
            else:
                if mode == 0:
                    x = torch.cat([x_other, x_focus_ori], dim=-1)
                elif mode == 1:
                    x = torch.cat([x_focus_ori[:,[0]], x_other, x_focus_ori[:,[1]]], dim=-1)
                else:
                    x = torch.cat([x_focus_ori, x_other], dim=-1)

        return x


# Deform
class TopoNetwork(nn.Module):
    def __init__(self,
                 d_feature,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 skip_in=(4,),
                 multires=0,
                 bias=0.5,
                 weight_norm=True,
                 isuse = True
                ):
        super(TopoNetwork, self).__init__()
        self.isuse = isuse
        dims_in = d_in
        dims = [d_in + d_feature] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embed_fn_fine = None

        if multires > 0:
            embed_fn, input_ch, _ = get_embedder(multires, input_dims=d_in)
            self.embed_fn_fine = embed_fn
            dims_in = input_ch
            dims[0] = input_ch + d_feature

        self.num_layers = len(dims)
        self.skip_in = skip_in

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims_in
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            
            setattr(self, "lin" + str(l), lin)
        
        self.activation = nn.ReLU()


    def forward(self, deformation_code, input_pts, alpha_ratio):
        if self.embed_fn_fine is not None:
            # Anneal
            input_pts = self.embed_fn_fine(input_pts)
        # x = torch.cat([input_pts, deformation_code.repeat(input_pts.shape[0],1)], dim=-1)
        x = torch.cat([input_pts, deformation_code], dim=-1)

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, input_pts], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)
        if self.isuse == False:
            x = torch.zeros_like(x).to(x)
        return x


class NGPNetwork(nn.Module):
    def __init__(self,
                 weight_norm=True,
                 multires_view=0,
                 squeeze_out=True):
        super().__init__()
        per_level_scale = np.exp2(np.log2(2048 * 1 / 16) / (16 - 1))

        self.encoder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": 8,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "per_level_scale": per_level_scale,
            },
        )
        dims = [16] + [32 for _ in range(2)] + [3]



        # latent code
        self.color_net = MLP_v1(c_in = dims[0], c_out=3, c_hiddens=dims, bn=None, zero_init=True)
        # self.color_net = tcnn.Network(
        #     n_input_dims=8,
        #     n_output_dims=3,
        #     network_config={
        #         "otype": "FullyFusedMLP",
        #         "activation": "ReLU",
        #         "output_activation": "None",
        #         "n_neurons": 64,
        #         "n_hidden_layers": 2,
        #     },
        # )



    def forward(self, points):
        color = self.color_net(self.encoder(points/np.pi))
        return color
    
class NGPNetwork_encode(nn.Module):
    def __init__(self,
                 weight_norm=True,
                 multires_view=0,
                 squeeze_out=True):
        super().__init__()
        per_level_scale = np.exp2(np.log2(2048 * 1 / 16) / (16 - 1))

        self.encoder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": 8,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "per_level_scale": per_level_scale,
            },
        )
        # dims = [16] + [32 for _ in range(2)] + [3]



        # latent code
        # self.color_net = MLP_v1(c_in = dims[0], c_out=3, c_hiddens=dims, bn=None, zero_init=True)
        self.color_net = tcnn.Network(
            n_input_dims=16,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 2,
            },
        )

        self.net = tcnn.NetworkWithInputEncoding(n_input_dims=3, n_output_dims=3,encoding=self.encoder, network=self.color_net)


    def forward(self, points):
        points = self.encoder(points/np.pi)
        output = self.color_net(points)
        return output

class AppearanceNetwork(nn.Module):
    def __init__(self,
                 d_feature,
                 d_global_feature,
                 mode,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 weight_norm=True,
                 multires_view=0,
                 squeeze_out=True):
        super().__init__()

        self.mode = mode
        self.squeeze_out = squeeze_out
        dims = [6  + d_feature + d_global_feature] + [d_hidden for _ in range(n_layers)] + [d_out]
        per_level_scale = np.exp2(np.log2(2048 * 1 / 16) / (16 - 1))

        # self.encoder = tcnn.Encoding(
        #     n_input_dims=3,
        #     encoding_config={
        #         "otype": "HashGrid",
        #         "n_levels": 16,
        #         "n_features_per_level": 2,
        #         "log2_hashmap_size": 19,
        #         "base_resolution": 16,
        #         "per_level_scale": per_level_scale,
        #     },
        # )
        self.embedview_fn = None
        if multires_view > 0:
            embedview_fn, input_ch, _ = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)

        self.color_net = MLP(in_ch= dims[0],
                            out_ch=3,
                            depth=n_layers,
                            hidden_activation=nn.ReLU(),
                            output_activation=None, 
                            width=d_hidden,
                            skips=[8,])


    def forward(self, global_feature, points, normals, view_dirs, feature_vectors, alpha_ratio, ngp_color=None):
        if self.embedview_fn is not None:
            # Anneal
            view_dirs = self.embedview_fn(view_dirs, alpha_ratio)

        rendering_input = None

        global_feature = global_feature#.repeat(points.shape[0],1)
        if self.mode == 'idr':
            rendering_input = torch.cat([view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_view_dir':
            rendering_input = torch.cat([normals, feature_vectors], dim=-1)
        elif self.mode == 'no_normal':
            rendering_input = torch.cat([view_dirs, feature_vectors], dim=-1)
        # points = self.encoder(points/3)
        # points = torch.zeros_like(points).to(points)
        # x =  torch.cat([rendering_input, points], dim=-1)
        x = rendering_input
        x = self.color_net(x)
       
        x = torch.sigmoid(x)
        return x



class SDFNetwork(nn.Module):
    def __init__(self,
                 d_in_1,
                 d_in_2,
                 d_out,
                 d_hidden,
                 n_layers,
                 skip_in=(4,),
                 multires=0,
                 multires_topo=0,
                 bias=0.5,
                 scale=1,
                 geometric_init=True,
                 weight_norm=True,
                 inside_outside=False):
        super(SDFNetwork, self).__init__()

        dims = [d_in_1 + d_in_2] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embed_fn_fine = None
        self.embed_amb_fn = None

        input_ch_1 = d_in_1
        input_ch_2 = d_in_2
        if multires > 0:
            embed_fn, input_ch_1 = get_embedder(multires, input_dims=d_in_1)
            self.embed_fn_fine = embed_fn
            dims[0] += (input_ch_1 - d_in_1)
        if multires_topo > 0:
            embed_amb_fn, input_ch_2 = get_embedder(multires_topo, input_dims=d_in_2)
            self.embed_amb_fn = embed_amb_fn
            dims[0] += (input_ch_2 - d_in_2)

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.scale = scale

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, d_in_1:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :d_in_1], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    if multires > 0:
                        torch.nn.init.constant_(lin.weight[:, -(dims[0] - d_in_1):-input_ch_2], 0.0)
                    if multires_topo > 0:
                        torch.nn.init.constant_(lin.weight[:, -(input_ch_2 - d_in_2):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.activation = nn.Softplus(beta=100)


    def forward(self, input_pts, topo_coord, alpha_ratio):
        input_pts = input_pts * self.scale
        if self.embed_fn_fine is not None:
            # Anneal
            input_pts = self.embed_fn_fine(input_pts, alpha_ratio)
        if self.embed_amb_fn is not None:
            # Anneal
            topo_coord = self.embed_amb_fn(topo_coord, alpha_ratio)
        inputs = torch.cat([input_pts, topo_coord], dim=-1)
        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)
        return torch.cat([x[:, :1] / self.scale, x[:, 1:]], dim=-1)


    # Anneal
    def sdf(self, x, topo_coord, alpha_ratio):
        return self.forward(x, topo_coord, alpha_ratio)[:, :1]


    def sdf_hidden_appearance(self, x, topo_coord, alpha_ratio):
        return self.forward(x, topo_coord, alpha_ratio)


    def gradient(self, x, topo_coord, alpha_ratio):
        x.requires_grad_(True)
        y = self.sdf(x, topo_coord, alpha_ratio)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients


# This implementation is based upon IDR: https://github.com/lioryariv/idr
class RenderingNetwork(nn.Module):
    def __init__(self,
                 d_feature,
                 mode,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 weight_norm=True,
                 multires_view=0,
                 squeeze_out=True):
        super().__init__()

        self.mode = mode
        self.squeeze_out = squeeze_out
        dims = [d_in + d_feature] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embedview_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()


    def forward(self, points, normals, view_dirs, feature_vectors):
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)

        rendering_input = None

        if self.mode == 'idr':
            rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_view_dir':
            rendering_input = torch.cat([points, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_normal':
            rendering_input = torch.cat([points, view_dirs, feature_vectors], dim=-1)

        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        if self.squeeze_out:
            x = torch.sigmoid(x)
        return x



class SingleBetaNetwork(nn.Module):
    def __init__(self, init_val):
        super(SingleBetaNetwork, self).__init__()
        ln_beta_init = np.log(init_val) 
        self.ln_beta = nn.Parameter(data=torch.Tensor([ln_beta_init]), requires_grad=True)
        
    def forward(self,):
        return   torch.exp(self.ln_beta)


class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val):
        super(SingleVarianceNetwork, self).__init__()
        self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))

    def forward(self, x):
        return torch.ones([len(x), 1]) * torch.exp(self.variance * 10.0)
