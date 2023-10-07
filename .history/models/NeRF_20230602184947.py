import torch
import torch.nn as nn
import numpy as np
from models.embedder import get_embedder
# import tinycudann as tcnn
import torch.nn.functional as F
from collections import OrderedDict
from functools import partial


class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a 
    # hyperparameter.
    
    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
         
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
    
    def forward_with_intermediate(self, input): 
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate
    
    
class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, 
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output # , coords        

    def forward_with_activations(self, coords, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!'''
        activations = OrderedDict()

        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)
                
                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()
                    
                activations['_'.join((str(layer.__class__), "%d" % activation_count))] = intermed
                activation_count += 1
            else: 
                x = layer(x)
                
                if retain_grad:
                    x.retain_grad()
                    
            activations['_'.join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return activations
    
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
            embed_fn, input_ch_1, _ = get_embedder(multires, input_dims=d_in_1)
            self.embed_fn_fine = embed_fn
            self.in_ch += (input_ch_1 - d_in_1)
        if multires_topo > 0:
            embed_amb_fn, input_ch_2, amb = get_embedder(multires_topo, use_input = False, use_alpha = True, input_dims=d_in_2)
            self.embed_amb_fn = embed_amb_fn
            self.in_ch += (input_ch_2 - d_in_2)
            self.amb = amb

            
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

        embed_rgb_fn, input_ch_3, _ = get_embedder(4, input_dims=3)
        self.embed_rgb_fn = embed_rgb_fn
        self.rgb_condition_dim += input_ch_3
        self.app_mlp = Siren(in_features=155,out_features=155,hidden_features=256,hidden_layers=3,outermost_linear=True)
        # # self.app_mlp = appNet(in_ch=158,out_ch=155)
        # self.app_bottleneck = nn.Linear(158,155)
        self.app_bottlneck = SineLayer(in_features=155,out_features=155)
        self.rgb_mlp = MLP(in_ch=self.rgb_branch_width+self.rgb_condition_dim,
                            out_ch=self.rgb_channels,
                            depth=self.rgb_branch_depth,
                            hidden_activation=self.hidden_activation,
                            output_activation=self.rgb_activation, 
                            width=self.rgb_branch_width,
                            skips=None)
        self.app_layer = SineLayer(in_features=3,out_features=3)
        # self.app_mlp_after = MLP(in_ch=3, out_ch=3,depth=self.rgb_branch_depth,
        #                     hidden_activation=self.hidden_activation,
        #                     output_activation=self.rgb_activation, 
        #                     width=self.rgb_branch_width,
        #                     skips=None)
        # self.app_Conv1d = nn.Conv1d(in_channels=800, out_channels=800, kernel_size=5, padding=4)
        self.alpha_mlp = nn.Linear(self.alpha_branch_width, self.alpha_channels)
        nn.init.xavier_uniform_(self.alpha_mlp.weight)
        
    def broadcast_condition(self,c,num_samples):
        # Broadcast condition from [batch, feature] to
        # [batch, num_coarse_samples, feature] since all the samples along the,
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
            input_pts = self.embed_fn_fine(x)
        if self.embed_amb_fn is not None:
            # Anneal
            topo_coord = self.embed_amb_fn(topo_coord)
        if self.embed_rgb_fn is not None:
            rgb_embed = self.embed_rgb_fn(dirs_o)
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
        # rgb_condition = rgb_condition.squeeze(1)
        if rgb_condition is not None:
            # rgb_condition = self.broadcast_condition(rgb_condition,x.shape[1])
            rgb_input = torch.cat([bottleneck, rgb_condition, rgb_embed],dim=-1)
            # rgb_input = self.app_mlp(rgb_input)
            rgb_input = self.app_bottleneck(rgb_input)
        else:
            rgb_input = torch.cat([bottleneck, rgb_embed],dim=-1)
        # rgb_input = self.app_mlp(rgb_input)
        # rgb_input = self.app_bottlneck(rgb_input)
        rgb = self.rgb_mlp(rgb_input)
        # rgb = self.siren_layer(rgb)
        # rgb = rgb.reshape(800,128,3)  # batchSize * nSample * 3
        # rgb = rgb.permute(1,0,2)
        # rgb = self.app_Conv1d(rgb)
        # rgb = rgb.permute(1,0,2)
        # rgb.reshape(102400,3)
        # rgb = self.app_mlp_after(rgb)
        return {'rgb':rgb,'alpha':alpha}


class NerfMLPpp(nn.Module):
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

        super(NerfMLPpp, self).__init__()
        input_ch_1 = d_in_1
        input_ch_2 = d_in_2

        self.in_ch = d_in_1 + d_in_2
        if multires > 0:
            embed_fn, input_ch_1, _ = get_embedder(multires, input_dims=d_in_1)
            self.embed_fn_fine = embed_fn
            self.in_ch += (input_ch_1 - d_in_1)
        if multires_topo > 0:
            embed_amb_fn, input_ch_2, amb = get_embedder(multires_topo, use_input = False, use_alpha = True, input_dims=d_in_2)
            self.embed_amb_fn = embed_amb_fn
            self.in_ch += (input_ch_2 - d_in_2)
            self.amb = amb

            
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

        embed_rgb_fn, input_ch_3, _ = get_embedder(4, input_dims=3)
        self.embed_rgb_fn = embed_rgb_fn
        self.rgb_condition_dim += input_ch_3

        self.rgb_mlp_pp = MLP(in_ch=self.rgb_branch_width+self.rgb_condition_dim,
                            out_ch=self.rgb_channels,
                            depth=self.rgb_branch_depth*2,
                            hidden_activation=self.hidden_activation,
                            output_activation=self.rgb_activation, 
                            width=self.rgb_branch_width*2,
                            skips=None)

        self.alpha_mlp = nn.Linear(self.alpha_branch_width, self.alpha_channels)
        nn.init.xavier_uniform_(self.alpha_mlp.weight)
        
    def broadcast_condition(self,c,num_samples):
        # Broadcast condition from [batch, feature] to
        # [batch, num_coarse_samples, feature] since all the samples along the,
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
            input_pts = self.embed_fn_fine(x)
        if self.embed_amb_fn is not None:
            # Anneal
            topo_coord = self.embed_amb_fn(topo_coord)
        if self.embed_rgb_fn is not None:
            rgb_embed = self.embed_rgb_fn(dirs_o)
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
        # rgb_condition = rgb_condition.squeeze(1)
        if rgb_condition is not None:
            # rgb_condition = self.broadcast_condition(rgb_condition,x.shape[1])
            rgb_input = torch.cat([bottleneck, rgb_condition, rgb_embed],dim=-1)
            # rgb_input = self.app_mlp(rgb_input)
            rgb_input = self.app_bottleneck(rgb_input)
        else:
            rgb_input = torch.cat([bottleneck, rgb_embed],dim=-1)

        rgb = self.rgb_mlp_pp(rgb_input)

        return {'rgb':rgb,'alpha':alpha}

   
class NerfSiren(nn.Module):
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

        super(NerfSiren, self).__init__()
        input_ch_1 = d_in_1
        input_ch_2 = d_in_2

        self.in_ch = d_in_1 + d_in_2
        if multires > 0:
            embed_fn, input_ch_1, _ = get_embedder(multires, input_dims=d_in_1)
            self.embed_fn_fine = embed_fn
            self.in_ch += (input_ch_1 - d_in_1)
        if multires_topo > 0:
            embed_amb_fn, input_ch_2, amb = get_embedder(multires_topo, use_input = False, use_alpha = True, input_dims=d_in_2)
            self.embed_amb_fn = embed_amb_fn
            self.in_ch += (input_ch_2 - d_in_2)
            self.amb = amb

            
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

        embed_rgb_fn, input_ch_3, _ = get_embedder(4, input_dims=3)
        self.embed_rgb_fn = embed_rgb_fn
        self.rgb_condition_dim += input_ch_3
        self.app_mlp = Siren(in_features=155,out_features=155,hidden_features=256,hidden_layers=3,outermost_linear=True)
        # # self.app_mlp = appNet(in_ch=158,out_ch=155)
        # self.app_bottleneck = nn.Linear(158,155)

        self.rgb_siren = Siren(in_features= self.rgb_branch_width+self.rgb_condition_dim, out_features=self.rgb_channels,hidden_features=256,hidden_layers=self.rgb_branch_depth,outermost_linear=True)

        self.alpha_mlp = nn.Linear(self.alpha_branch_width, self.alpha_channels)
        nn.init.xavier_uniform_(self.alpha_mlp.weight)
        
    def broadcast_condition(self,c,num_samples):
        # Broadcast condition from [batch, feature] to
        # [batch, num_coarse_samples, feature] since all the samples along the,
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
            input_pts = self.embed_fn_fine(x)
        if self.embed_amb_fn is not None:
            # Anneal
            topo_coord = self.embed_amb_fn(topo_coord)
        if self.embed_rgb_fn is not None:
            rgb_embed = self.embed_rgb_fn(dirs_o)
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
        # rgb_condition = rgb_condition.squeeze(1)
        if rgb_condition is not None:
            # rgb_condition = self.broadcast_condition(rgb_condition,x.shape[1])
            rgb_input = torch.cat([bottleneck, rgb_condition, rgb_embed],dim=-1)
            # rgb_input = self.app_mlp(rgb_input)
            rgb_input = self.app_bottleneck(rgb_input)
        else:
            rgb_input = torch.cat([bottleneck, rgb_embed],dim=-1)

        rgb = self.rgb_siren(rgb_input)
        rgb = self.rgb_activation(rgb)
        return {'rgb':rgb,'alpha':alpha}


class MLP_v1(nn.Module):
    def __init__(self, c_in, c_out, c_hiddens, act=nn.LeakyReLU, bn=nn.BatchNorm1d, zero_init=False):
        super().__init__()
        layers = []
        d_in = c_in
        for d_out in c_hiddens:
            layers.append(nn.Linear(d_in, d_out)) # nn.Conv1d(d_in, d_out, 1, 1, 0)
            if bn is not None:
                layers.append(bn(d_out))
            layers.append(act())
            d_in = d_out
        layers.append(nn.Linear(d_in, c_out)) # nn.Conv1d(d_in, c_out, 1, 1, 0)
        if zero_init:
            nn.init.constant_(layers[-1].bias, 0.0)
            nn.init.constant_(layers[-1].weight, 0.0)
        self.mlp = nn.Sequential(*layers)
        self.c_out = c_out


    def forward(self, x):
        x = x.float()
        return torch.tanh(self.mlp(x))

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

class ResBlock(nn.Module):

    def __init__(self,c):
        super(ResBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(c,c,3,1,1, bias=False),
            nn.InstanceNorm2d(c),
            nn.ReLU(),
            nn.Conv2d(c, c, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(c),

        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.layer(x)+x)

class appNet(nn.Module):

    def __init__(self,in_ch,out_ch):
        super(appNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm1d(out_ch*2),
            nn.Sigmoid(),
            nn.Conv1d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm1d(out_ch*2),
            nn.Sigmoid()
        )


    def forward(self, x):
        return self.net(x)

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

