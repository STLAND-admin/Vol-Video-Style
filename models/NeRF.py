import torch
import torch.nn as nn
import numpy as np
from models.embedder import get_embedder
# import tinycudann as tcnn

from functools import partial

    
class NerfMLP(nn.Module):
    """A simple MLP.
    Attributes:
        nerf_trunk_depth: int, the depth of the first part of MLP.
        nerf_trunk_width: int, the width of the first part of MLP.
        nerf_rgb_branch_depth: int, the depth of the second part of MLP.
        nerf_rgb_branch_width: int, the width of the second part of MLP.
        activation: function, the activation function used in the MLP.
        skips: which layers to add skip layers to.
        alpha_channels: int, the number of alpha_channelss.
        rgb_channels: int, the number of rgb_channelss.
        condition_density: if True put the condition at the begining which
        conditions the density of the field.
    """

    def __init__(self,d_in_1, d_in_2, multires, multires_topo,
                trunk_depth,
                trunk_width,
                rgb_branch_depth,
                rgb_branch_width,
                alpha_channels,
                rgb_channels,
                norm=None,
                skips=[4,],
                alpha_brach_depth=1,
                alpha_brach_width=128,
                rgb_condition_dim = 0
                ):

        super(NerfMLP, self).__init__()
        input_ch_1 = d_in_1
        input_ch_2 = d_in_2

        self.in_ch = d_in_1 + d_in_2
        if multires > 0:
            embed_fn, input_ch_1 = get_embedder(multires, input_dims=d_in_1)
            self.embed_fn_fine = embed_fn
            self.in_ch += (input_ch_1 - d_in_1)
        if multires_topo > 0:
            embed_amb_fn, input_ch_2 = get_embedder(multires_topo, input_dims=d_in_2)
            self.embed_amb_fn = embed_amb_fn
            self.in_ch += (input_ch_2 - d_in_2)
            

            
        self.trunk_depth = trunk_depth
        self.trunk_width = trunk_width
        self.rgb_branch_depth = rgb_branch_depth
        self.rgb_branch_width = rgb_branch_width
        self.rgb_channels = rgb_channels
        self.alpha_branch_depth = alpha_brach_depth
        self.alpha_branch_width = alpha_brach_width
        self.alpha_channels = alpha_channels
        self.rgb_condition_dim = rgb_condition_dim
        self.condition_density = False
        if skips is None:
            self.skips = [4,]
        self.skips = skips

        self.hidden_activation = nn.ReLU()

        # if rgb_activation == None:
        #     self.rgb_activation = nn.Identity()
        # elif rgb_activation == 'sigmoid':
        self.rgb_activation = nn.Sigmoid()

        self.sigma_activation = nn.Identity()

        self.norm = norm

        #todo check this
        self.trunk_mlp = MLP(in_ch=self.in_ch,
                            out_ch=self.trunk_width,
                            depth=self.trunk_depth,
                            width=self.trunk_width,
                            hidden_activation=self.hidden_activation,
                            skips=self.skips,
                            output_activation=self.hidden_activation)

        self.bottleneck_mlp = nn.Linear(self.trunk_width,self.trunk_width//2)#128

        embed_rgb_fn, input_ch_3 = get_embedder(4, input_dims=3)
        self.embed_rgb_fn = embed_rgb_fn
        self.rgb_condition_dim += input_ch_3
        
        self.rgb_mlp = MLP(in_ch=self.rgb_branch_width+self.rgb_condition_dim,
                            out_ch=self.rgb_channels,
                            depth=self.rgb_branch_depth,
                            hidden_activation=self.hidden_activation,
                            output_activation=self.rgb_activation, 
                            width=self.rgb_branch_width,
                            skips=self.skips)


        self.alpha_mlp = nn.Linear(self.alpha_branch_width, self.alpha_channels)
        nn.init.xavier_uniform_(self.alpha_mlp.weight)
        
    def broadcast_condition(self,c,num_samples):
        # Broadcast condition from [batch, feature] to
        # [batch, num_coarse_samples, feature] since all the samples along the
        # same ray has the same viewdir.
        if c.dim() == 2:
            c = c.unsqueeze(1)
        c = c.repeat(1,num_samples,1)        
        # Collapse the [batch, num_coarse_samples, feature] tensor to
        # [batch * num_coarse_samples, feature] to be fed into nn.Dense.
        # c = c.view([-1, c.shape[-1]])
        return c

    def forward(self,x, topo_coord, dirs_o, rgb_condition= None, alpha_ratio=None):
        """
            Args:
            x: sample points with shape [batch, num_coarse_samples, feature].
            alpha_condition: a condition array provided to the alpha branch.
            rgb_condition: a condition array provided in the RGB branch.
            Returns:
            raw: [batch, num_coarse_samples, rgb_channels+alpha_channels].
        """
        if self.embed_fn_fine is not None:
            # Anneal
            input_pts = self.embed_fn_fine(x, alpha_ratio)
        if self.embed_amb_fn is not None:
            # Anneal
            topo_coord = self.embed_amb_fn(topo_coord, alpha_ratio)
        if self.embed_rgb_fn is not None:
            rgb_embed = self.embed_rgb_fn(dirs_o, alpha_ratio)
        inputs = torch.cat([input_pts, topo_coord], dim=-1)
        x = self.trunk_mlp(inputs)
        # bottleneck = self.bottleneck_mlp(x)
        bottleneck = self.bottleneck_mlp(x)

        # if alpha_condition is not None:
        #     # alpha_condition = self.broadcast_condition(alpha_condition,x.shape[1])
        #     alpha_condition = alpha_condition
        #     alpha_input = torch.cat([bottleneck,alpha_condition],dim=-1)
        # else:
        alpha_input = bottleneck


        alpha = self.alpha_mlp(alpha_input)
        rgb_condition = None
        if rgb_condition is not None:
            # rgb_condition = self.broadcast_condition(rgb_condition,x.shape[1])
            rgb_input = torch.cat([bottleneck, rgb_condition, rgb_embed],dim=-1)
        else:
            rgb_input = torch.cat([bottleneck, rgb_embed],dim=-1)
        rgb = self.rgb_mlp(rgb_input)

        return {'rgb':rgb,'alpha':alpha}

class MLP(nn.Module):
    """A multi-layer perceptron.
    Args:
        in_channels: The number of input channels.
        out_channels: The number of output channels.
        hidden_channels: The number of hidden channels.
        hidden_layers: The number of hidden layers.
        hidden_activation: The activation function for the hidden layers.
        hidden_norm: A string indicating the type of norm to use for the hidden
            layers.
        out_activation: The activation function for the output layer.
        out_norm: A string indicating the type of norm to use for the output
            layer.
        dropout: The dropout rate.
    """

    def __init__(self,in_ch:int,out_ch:int, depth:int=8,width:int=256,hidden_init=None,hidden_activation=None,
            hidden_norm=None,output_init=None, output_activation=None,
            use_bias=True,skips=None):
        super(MLP, self).__init__() 
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.depth = depth
        self.width = width
        if hidden_init is None:
            self.hidden_init = nn.init.xavier_uniform_
        else:
            self.hidden_init = hidden_init

        if hidden_activation == None:
            self.hidden_activation = nn.ReLU()
        else:
            self.hidden_activation = hidden_activation

        self.hidden_norm = hidden_norm

        if output_init is None:
            self.output_init = nn.init.xavier_uniform_
        else:
            self.output_init = output_init

        if output_activation == None:
            self.output_activation = nn.Identity()
        else:
            self.output_activation = output_activation

        self.use_bias = use_bias
        if skips is None:
            self.skips = [4,]
        else:
            self.skips = skips

        self.linears = nn.ModuleList([nn.Linear(in_ch, width)] + 
            [nn.Linear(width, width) if i not in self.skips else 
            nn.Linear(width+ in_ch, width) for i in range(depth-1)])
        self.logit_layer = nn.Linear(width, out_ch)

        # initalize using glorot
        for _, linear in enumerate(self.linears):
            self.hidden_init(linear.weight)
        # initialize output layer
        if self.output_init is not None:
            self.output_init(self.logit_layer.weight)

  

    def forward(self,inputs):
        x = inputs
        for i, linear in enumerate(self.linears):
            x = linear(x)
            x = self.hidden_activation(x)
            # if self.hidden_norm is not None:
            #     x = self.norm_layers[i](x)
            if i in self.skips:
                x = torch.cat([x,inputs],-1)
        x = self.logit_layer(x)
        x = self.output_activation(x)
        return x


# class NeRFNetwork(nn.Module):
#     def __init__(self,
#                  d_in_1,
#                  d_in_2,
#                  d_out,
#                  d_hidden,
#                  n_layers,
#                  skip_in=(4,),
#                  multires=0,
#                  multires_topo=0,
#                  bias=0.5,
#                  scale=1,
#                  geometric_init=True,
#                  weight_norm=True,
#                  inside_outside=False):
#         super(NeRFNetwork, self).__init__()

#         dims = [d_in_1 + d_in_2] + [d_hidden for _ in range(n_layers)] + [d_out]

#         self.embed_fn_fine = None
#         self.embed_amb_fn = None

#         input_ch_1 = d_in_1
#         input_ch_2 = d_in_2
#         if multires > 0:
#             embed_fn, input_ch_1 = get_embedder(multires, input_dims=d_in_1)
#             self.embed_fn_fine = embed_fn
#             dims[0] += (input_ch_1 - d_in_1)
#         if multires_topo > 0:
#             embed_amb_fn, input_ch_2 = get_embedder(multires_topo, input_dims=d_in_2)
#             self.embed_amb_fn = embed_amb_fn
#             dims[0] += (input_ch_2 - d_in_2)

#         self.num_layers = len(dims)
#         self.skip_in = skip_in
#         self.scale = scale

#         for l in range(0, self.num_layers - 1):
#             if l + 1 in self.skip_in:
#                 out_dim = dims[l + 1] - dims[0]
#             else:
#                 out_dim = dims[l + 1]

#             lin = nn.Linear(dims[l], out_dim)

         

#             if weight_norm:
#                 lin = nn.utils.weight_norm(lin)

#             setattr(self, "lin" + str(l), lin)

#         self.activation = nn.ReLU()


#     def forward(self, input_pts, topo_coord, alpha_ratio):
#         input_pts = input_pts * self.scale
#         if self.embed_fn_fine is not None:
#             # Anneal
#             input_pts = self.embed_fn_fine(input_pts, alpha_ratio)
#         if self.embed_amb_fn is not None:
#             # Anneal
#             topo_coord = self.embed_amb_fn(topo_coord, alpha_ratio)
#         inputs = torch.cat([input_pts, topo_coord], dim=-1)
#         x = inputs
#         for l in range(0, self.num_layers - 1):
#             lin = getattr(self, "lin" + str(l))

#             if l in self.skip_in:
#                 x = torch.cat([x, inputs], 1) / np.sqrt(2)

#             x = lin(x)

#             if l < self.num_layers - 2:
#                 x = self.activation(x)
#         return x

