import torch
import math
import torch.nn as nn
import math
torch.pi = math.pi

# Anneal, Coarse-to-Fine Optimization part proposed by:
# Park, Keunhong, et al. Nerfies: Deformable neural radiance fields. CVPR 2021.
class Embedder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.create_embedding_fn()
        self.use_alpha = kwargs['use_alpha']
        if self.use_alpha == True:
            self.register_parameter('alpha' , nn.Parameter(torch.zeros(1,)))
            self.alpha.require_grad = False 
        self.use_input = kwargs['use_input']
        if self.use_input:
            self.out_dim += 3
    def create_embedding_fn(self):
        embed_fns = []
        self.input_dims = self.kwargs['input_dims']
        out_dim = 0
        self.include_input = self.kwargs['include_input']

        max_freq = self.kwargs['max_freq_log2']
        self.num_freqs = self.kwargs['num_freqs']
        out_dim += self.input_dims * self.num_freqs * 2

        self.freq_bands = 2. ** torch.linspace(0., max_freq, self.num_freqs) 

        self.num_fns = len(self.kwargs['periodic_fns'])
  

        self.embed_fns = embed_fns
        self.out_dim = out_dim
       
    # Anneal. Initial alpha value is 0, which means it does not use any PE (positional encoding)!
    def embed(self, inputs):
    # (..., F, C).
        batch_shape = inputs.shape[:-1]

        xb = inputs[..., None, :] * self.freq_bands[:, None]
        # (..., F, 2, C).
        four_feats = torch.sin(torch.stack([xb, xb + 0.5 * math.pi], axis=-2))

        if self.use_alpha:
            window = _posenc_window(self.num_freqs, self.alpha)
            four_feats = window[..., None, None] * four_feats
        four_feats = four_feats.reshape((*batch_shape, -1))

        if self.use_input:
            output = torch.cat([inputs, four_feats], axis=-1)
        else:
            output = four_feats
        return output

def _posenc_window(num_freqs: int, alpha):
    """Windows a posenc using a cosiney window.

    This is equivalent to taking a truncated Hann window and sliding it to the
    right along the frequency spectrum.

    Args:
        num_freqs (int): The number of frequencies in the posenc.
        alpha (jnp.ndarray): The maximal frequency that allows by the window.

    Returns:
        jnp.ndarray: A (..., num_freqs) array of window values.
    """
    freqs = torch.arange(num_freqs).float()
    xs = torch.clip(alpha - freqs, 0, 1)
    return 0.5 * (1 + torch.cos(torch.pi * xs + torch.pi))

def get_embedder(multires, use_input = True, use_alpha = False, input_dims=3):
    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dims,
        'max_freq_log2': multires-1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
        'use_alpha': use_alpha,
        'use_input': use_input
    }

    embedder_obj = Embedder(**embed_kwargs)
    def embed(x, eo=embedder_obj): return eo.embed(x)
    return embed, embedder_obj.out_dim, embedder_obj

