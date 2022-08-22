import torch
import einops
from einops.layers.torch import Rearrange
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import MDAnalysis as mda
import warnings
import gc
import inspect
import curtsies.fmtfuncs as fmt
from typing import *
import cv2
import tqdm 

warnings.simplefilter("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_printoptions(precision=4)

class Encoder(torch.nn.Module):
    def __init__(self, dim=[128,64,4], **kwargs):
        super().__init__()
        self.dim = dim
        self.voc_size = kwargs.get("n_e")
        image_shape = kwargs.get("image_shape") #min_encoding_size: int
        print(image_shape)
        convs = torch.nn.Sequential(*[ 
                                      torch.nn.Conv2d(1,64,kernel_size=3), torch.nn.AdaptiveAvgPool2d(int(image_shape*0.9)), torch.nn.LeakyReLU(), 
                                      torch.nn.Conv2d(64,128,kernel_size=2), torch.nn.AdaptiveAvgPool2d(int(image_shape*0.7)),  torch.nn.LeakyReLU(), 
                                      torch.nn.Conv2d(128,128,kernel_size=2), torch.nn.AdaptiveAvgPool2d(int(image_shape*0.4)),  torch.nn.LeakyReLU(),
                                      torch.nn.Conv2d(128,128,kernel_size=2), torch.nn.AdaptiveAvgPool2d(int(image_shape*0.2)),  torch.nn.LeakyReLU(),
                                      torch.nn.Conv2d(128,128,kernel_size=2), torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(1,-1)
                                    ]) #B,C,H,W
        self.add_module("conv_sequential", convs)
        lins = torch.nn.Sequential(*list(itertools.chain(*[[torch.nn.Linear(dim[d], dim[d+1]), torch.nn.Dropout(p=0.5)] for d in range(len(dim)-1) 
                                ])))
        self.add_module("lin_sequential", lins)
    def forward(self, inputs):
        x = inputs #BHW -> Coded indices
        sizes = inputs.size()
        x = x.view(sizes[0],1,sizes[1],sizes[2])
        x = self.normalize(x)
        x = self.conv_sequential(x)
        x = self.lin_sequential(x)
        mu, logstd = torch.chunk(x, 2, dim=-1)
        z = self.reparameterize(mu, logstd)
        return z, mu, logstd
    def reparameterize(self, mu, logstd):
        shapes = mu.shape
        # print(shapes)
        return mu + logstd.exp() * torch.distributions.Normal(0., 0.3).rsample((shapes)).to(device)
    def normalize(self, image: "Integer tensor"):
        image = image / (self.voc_size - 1)
        return image

# enc = Encoder(n_e=100)
# z, _, _ = enc(min_encoding_indices)
# plt.scatter(*z.detach().T)

class Decoder(torch.nn.Module):
    def __init__(self, dim=list(reversed([128,64,2])), **kwargs):
        super().__init__()
        self.dim = dim
        self.voc_size = kwargs.get("n_e")
        image_shape = kwargs.get("image_shape")
        lins = torch.nn.Sequential(*list(itertools.chain(*[[torch.nn.Linear(dim[d], dim[d+1]), torch.nn.Dropout(p=0.5)] for d in range(len(dim)-1) 
                                ])))
        self.add_module("lin_sequential", lins)
        convs = torch.nn.Sequential(*[
torch.nn.ConvTranspose2d(128,128, kernel_size=2), torch.nn.Upsample(int(image_shape*0.2), mode='bilinear'), torch.nn.LeakyReLU(), 
torch.nn.ConvTranspose2d(128,128, kernel_size=2), torch.nn.Upsample(int(image_shape*0.4), mode='bilinear'), torch.nn.LeakyReLU(), 
torch.nn.ConvTranspose2d(128,128, kernel_size=2), torch.nn.Upsample(int(image_shape*0.7), mode='bilinear'), torch.nn.LeakyReLU(), 
torch.nn.ConvTranspose2d(128,64, kernel_size=2), torch.nn.Upsample(int(image_shape*0.9), mode='bilinear'),  torch.nn.LeakyReLU(), 
torch.nn.ConvTranspose2d(64,1,kernel_size=3), torch.nn.Upsample(image_shape, mode='bilinear'), 
torch.nn.Sigmoid()
                            ]) #B,C,H,W
        self.add_module("conv_sequential", convs)
    def forward(self, inputs: "BD"):
        x = inputs
        x = self.lin_sequential(x)
        sizes = x.size()
        x = x.view(sizes[0],sizes[1],1,1) #B,C,1,1
        x = self.conv_sequential(x)
        x_ = self.unnormalize(x) #Back to integer code indices; not for training...; B1HW
        return x, x_.detach().squeeze(1)
    def unnormalize(self, image: "Sigmoided tensor"):
        image = (image) * (self.voc_size - 1)
        return image    
# dec = Decoder(n_e=100, image_shape=min_encoding_indices.size(1))
# x, x_ = dec(z)

class VAE(torch.nn.Module):
    """VAE github: https://github.com/AntixK/PyTorch-VAE/tree/master/models"""
    def __init__(self, **kwargs):
        super().__init__()
        voc_size = kwargs.get("n_e")
        image_shape = kwargs.get("image_shape")
        self.enc = Encoder(n_e=voc_size, image_shape=image_shape)
        self.dec = Decoder(n_e=voc_size, image_shape=image_shape)
        # self.apply(self._init_weights)
        self.extract_features()
    def forward(self, inputs):
        x = inputs
        en, mu, logstd = self.enc(x)
        x, image = self.dec(en)
        return x, en, mu, logstd, image
    def losses(self, inputs, recon, z, mu, logstd):
        bce = torch.nn.BCELoss()(recon, inputs)
        #mse = torch.nn.MSELoss(reduction="mean")(recon, inputs).mean()
        #kl = torch.distributions.Normal(mu, logstd.exp()).log_prob(z) - torch.distributions.Normal(0., 1.).log_prob(z)
        kl = torch.mean(-0.5 * torch.sum(1 + logstd - mu ** 2 - logstd.exp(), dim = 1), dim = 0)

        feature_num = ['2','5','8','11']
        feat = 0.
        for f, b in zip(feature_num, list(reversed(feature_num))):
            fvec = self.enc_dict.get(f)
            bvec = self.dec_dict.get(b)
            feat = feat + torch.nn.MSELoss(reduction="mean")(fvec, bvec)
        return bce + kl + feat
    def extract_features(self, ):
        """
        Extracts the features from the pretrained model
        at the layers indicated by feature_layers.
        :param input: (Tensor) [B x C x H x W]
        :param feature_layers: List of string of IDs
        :return: List of the extracted features
        """
        feature_num = ['2','5','8','11'] #Which module to extract: symmetric!
        self.enc_dict = dict()
        self.dec_dict = dict()
        def enc_hook_return(layer_name):
            def enc_hook(m, i, o):
                self.enc_dict[layer_name] = o
            return enc_hook
        def dec_hook_return(layer_name):
            def dec_hook(m, i, o):
                self.dec_dict[layer_name] = o
            return dec_hook
        for name, l in self.enc.conv_sequential.named_modules():
            if name in feature_num:
                l.register_forward_hook(enc_hook_return(name))
        for name, l in self.dec.conv_sequential.named_modules():
            if name in feature_num:
                l.register_forward_hook(dec_hook_return(name))
    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear) and hasattr(m, 'weight') and m.weight is not None:
            torch.nn.init.uniform_(m.weight,0,1)
        if isinstance(m, torch.nn.Linear) and hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0.)
        if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)) and hasattr(m, 'weight') and m.weight is not None:
            torch.nn.init.uniform_(m.weight)
        if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)) and hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0.)

class GaussianSmearing(torch.nn.Module):
    #Schnet
    def __init__(self,start=0.0, stop=5.0, n_gaussians=64, centered=False, trainable=False):
        super().__init__()
        offset = torch.linspace(start, stop, n_gaussians)
        widths = torch.FloatTensor((offset[1] - offset[0]) * torch.ones_like(offset))
        if trainable:
            self.register_parameter("width", torch.nn.Parameter(widths))
            self.register_parameter("offsets", torch.nn.Parameter(offset)) #1,1,1,C
        else:
            self.register_buffer("width", widths)
            self.register_buffer("offsets", offset)
        self.centered = centered
    def forward(self, distances: "distances"):
        # D = distances.unsqueeze(-1) if distances.ndim != 2 else distances
        assert distances.ndim == 4 and distances.size(1) == 1, "Something has gone awry..." 
        D = distances.permute(0,2,3,1) #Channel last...
        diff = D - self.offsets
        coeff = -0.5 / torch.pow(self.width, 2)
        gauss = torch.exp(coeff * torch.pow(diff, 2))
        return gauss.permute(0,3,1,2)

class RBFLayer_Physnet(torch.nn.Module):
    #Physnet
    def __init__(self, dfilter=64, cutoff=5, **kwargs):
        super().__init__(**kwargs)
        self.register_buffer('filter', torch.as_tensor(dfilter, dtype=torch.int32))
        self.register_buffer('cutoff', torch.as_tensor(cutoff, dtype=torch.float32))
        self.register_parameter("centers", torch.nn.Parameter(torch.nn.Softplus()(self._softplus_inverse(torch.linspace(1, torch.exp(-self.cutoff), self.filter))), requires_grad=True))
        self.register_parameter("widths", torch.nn.Parameter(torch.nn.Softplus()(torch.as_tensor([self._softplus_inverse((0.5/((1.0-torch.exp(-self.cutoff))/self.filter))**2)]*self.filter)), requires_grad=True))

    def _softplus_inverse(self, x: torch.Tensor):
        """Numerically stable"""
        return x + torch.log(-torch.expm1(-x + torch.finfo().eps))

    def _cutoff_fn(self, dist_expanded):
        #x = self.dist_expanded / self.cutoff
        #print("dist_expanded", self.dist_expanded.device, "cutoff", self.cutoff.device)
        x = dist_expanded / self.cutoff
        x3 = x**3
        x4 = x3*x
        x5 = x4*x
        return torch.where(x < 1, 1 - 6*x5 + 15*x4 - 10*x3, torch.zeros_like(x))

    def forward(self, distances):
        # D = distances.unsqueeze(-1) if distances.ndim != 2 else distances
        assert distances.ndim == 4 and distances.size(1) == 1, "Something has gone awry..." 
        #D = distances.permute(0,2,3,1) #Channel last...
        shape = distances.size()
        D = distances.view(-1,1) 
        cutoff_val = self._cutoff_fn(D)
        rbf = cutoff_val * torch.exp(-self.widths * (torch.exp(-D) - self.centers)**2)
        return rbf.contiguous().view(shape[0], self.filter.item(), shape[2], shape[3])

class VectorQuantizer(torch.nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """
    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(self, n_e: "Voc size", e_dim: "Emb dim", beta, unknown_index="random",
                  legacy=True):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.legacy = legacy

        self.embedding = torch.nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        self.re_embed = n_e

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp==1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits==False, "Only for interface compatible with Gumbel"
        assert return_logits==False, "Only for interface compatible with Gumbel"
        # reshape z -> (batch, height, width, channel) and flatten
        z = einops.rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, einops.rearrange(self.embedding.weight, 'n d -> d n'))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        perplexity = None
        min_encodings = None

        # compute loss for embedding
        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach()-z)**2) + \
                   torch.mean((z_q - z.detach()) ** 2)
        else:
            loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
                   torch.mean((z_q - z.detach()) ** 2)
        # preserve gradients
        z_q = z + (z_q - z).detach()
        # reshape back to match original input shape
        z_q = einops.rearrange(z_q, 'b h w c -> b c h w').contiguous()
        return z_q, loss, (perplexity, min_encodings, min_encoding_indices.view(-1, z.size(1), z.size(2)))

    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        if self.remap is not None:
            indices = indices.reshape(shape[0],-1) # add batch axis
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1) # flatten again
        # get quantized latent vectors
        z_q = self.embedding(indices)
        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

mList = torch.nn.ModuleList

# normalizations
class PreNorm(torch.nn.Module):
    def __init__(
        self,
        dim,
        fn
    ):
        super().__init__()
        self.fn = fn
        self.norm = torch.nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args,**kwargs)

# gated residual
class Residual(torch.nn.Module):
    def forward(self, x, res):
        return x + res

class GatedResidual(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(dim * 3, 1, bias = False),
            torch.nn.Sigmoid()
        )
    def forward(self, x, res):
        gate_input = torch.cat((x, res, x - res), dim = -1)
        gate = self.proj(gate_input)
        return x * gate + res * (1 - gate)

def rotate_half(x):
    x = einops.rearrange(x, '... (d r) -> ... d r', r = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return einops.rearrange(x, '... d r -> ... (d r)')

def apply_rotary_emb(freqs, t, start_index = 0):
    freqs = freqs.to(t)
    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim
    assert rot_dim <= t.shape[-1], f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'
    t_left, t, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]
    t = (t * freqs.cos()) + (rotate_half(t) * freqs.sin())
    return torch.cat((t_left, t, t_right), dim = -1)

class RotaryEmbedding(torch.nn.Module):
    def __init__(
        self,
        dim,
        custom_freqs = None,
        freqs_for = 'lang',
        theta = 10000,
        max_freq = 10,
        num_freqs = 1,
        learned_freq = False
    ):
        super().__init__()
        if exists(custom_freqs):
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1., max_freq / 2, dim // 2) * np.pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f'unknown modality {freqs_for}')

        self.cache = dict()

        if learned_freq:
            self.freqs = torch.nn.Parameter(freqs)
        else:
            self.register_buffer('freqs', freqs)

    def forward(self, t, cache_key = None):
        if exists(cache_key) and cache_key in self.cache:
            return self.cache[cache_key]

        if inspect.isfunction(t):
            t = t()

        freqs = self.freqs

        freqs = torch.einsum('..., f -> ... f', t.type(freqs.dtype), freqs)
        freqs = einops.repeat(freqs, '... n -> ... (n r)', r = 2)

        if exists(cache_key):
            self.cache[cache_key] = freqs

        return freqs

# attention
class Attention(torch.nn.Module):
    def __init__(
        self,
        dim,
        pos_emb = None,
        dim_head = 64,
        heads = 8,
        edge_dim = None,
        sequence_length = None
    ):
        super().__init__()
        edge_dim = default(edge_dim, 64)
        sequence_length = default(sequence_length, 200)

        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.pos_emb = pos_emb
        self.to_q = torch.nn.Linear(inner_dim, inner_dim)
        self.to_kv = torch.nn.Linear(inner_dim, inner_dim * 2)
        self.edges_to_kv = torch.nn.Linear(edge_dim, inner_dim)
        self.to_out = torch.nn.Linear(inner_dim, dim)
        #self.pos_emb = torch.nn.Embedding(sequence_length, dim) 
        self.node_expansion = torch.nn.Linear(dim, inner_dim)
    def forward(self, nodes, edges, mask = None):
        h = self.heads
        nodes = self.node_expansion(nodes)
        q = self.to_q(nodes)
        k, v = self.to_kv(nodes).chunk(2, dim = -1)
        # edges = edges.view(edges.size(0), edges.size(2), edges.size(3), -1)
        edges = edges.permute(0,2,3,1)
        e_kv = self.edges_to_kv(edges)
        q, k, v, e_kv = map(lambda t: einops.rearrange(t, 'b ... (h d) -> (b h) ... d', h = h), (q, k, v, e_kv))
        if exists(self.pos_emb):
            freqs = self.pos_emb(torch.arange(nodes.shape[1], device = nodes.device))
            freqs = einops.rearrange(freqs, 'n d -> () n d')
            q = apply_rotary_emb(freqs, q)
            k = apply_rotary_emb(freqs, k)
        
        ek, ev = e_kv, e_kv
        k, v = map(lambda t: einops.rearrange(t, 'b j d -> b () j d '), (k, v))
        k = k + ek
        v = v + ev
        sim = torch.einsum('b i d, b i j d -> b i j', q, k) * self.scale
        if exists(mask):
            mask = einops.rearrange(mask, 'b i -> b i ()') & einops.rearrange(mask, 'b j -> b () j')
            max_neg_value = -torch.finfo(sim.dtype).max
            sim.masked_fill_(~mask, max_neg_value)
        attn = sim.softmax(dim = -1)
        out = torch.einsum('b i j, b i j d -> b i d', attn, v)
        out = einops.rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)

# optional feedforward
def FeedForward(dim, ff_mult = 4):
    return torch.nn.Sequential(
        torch.nn.Linear(dim, dim * ff_mult),
        torch.nn.GELU(),
        torch.nn.Linear(dim * ff_mult, dim)
    )

# classes
class GraphTransformer(torch.nn.Module):
    def __init__(
        self,
        dim,
        depth,
        dim_head = 64,
        edge_dim = None,
        heads = 8,
        gated_residual = True,
        with_feedforwards = True,
        norm_edges = True,
        rel_pos_emb = True,
        sequence_length = None
    ):
        super().__init__()
        self.layers = mList([])
        edge_dim = default(edge_dim, 64)
        self.norm_edges = torch.nn.LayerNorm(edge_dim) if norm_edges else torch.nn.Identity()
        pos_emb = RotaryEmbedding(dim_head) if rel_pos_emb else None
        #See 1-3D pos-emb: https://github.com/tatp22/multidim-positional-encoding/blob/master/positional_encodings/positional_encodings.py
        for _ in range(depth):
            self.layers.append(mList([
                mList([
                    PreNorm(dim, Attention(dim, pos_emb = pos_emb, edge_dim = edge_dim, dim_head = dim_head, heads = heads, sequence_length=sequence_length)),
                    GatedResidual(dim)
                ]),
                mList([
                    PreNorm(dim, FeedForward(dim)),
                    GatedResidual(dim)
                ]) if with_feedforwards else None
            ]))
    def forward(self, nodes, edges, mask = None):
        edges = self.norm_edges(edges.permute(0,2,3,1)).permute(0,3,1,2)
        for attn_block, ff_block in self.layers:
            attn, attn_residual = attn_block
            nodes = attn_residual(attn(nodes, edges, mask = mask), nodes)
            if exists(ff_block):
                ff, ff_residual = ff_block
                nodes = ff_residual(ff(nodes), nodes)
        return nodes, edges

class DistanceMapping(torch.nn.Module):
    """Batch by atom/residue pair distance"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def forward(self, coordinates: "B,L,1"):
        x = coordinates
        src = x.repeat_interleave(x.size(1), dim=1) #B, L**2, 1
        tgt = x.repeat(1,x.size(1),1) #B, L**2, 3
        distmap = (src.view(-1,3) - tgt.view(-1,3)).pow(2).sum(dim=-1, keepdim=True).sqrt().view(x.size(0), x.size(1), x.size(1)) #B, L, L
        return distmap

class EncoderConv(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        image_shape = kwargs.get("image_shape") #H,W
        min_encoding_size = kwargs.get("min_encoding_size")
        picture_dim = kwargs.get("picture_dim") 
        convs = torch.nn.Sequential(*[ 
                                      torch.nn.Conv2d(picture_dim,64,kernel_size=3), torch.nn.AdaptiveAvgPool2d(int(image_shape[0]*0.95)), torch.nn.LeakyReLU(), 
                                      torch.nn.Conv2d(64,128,kernel_size=2), torch.nn.AdaptiveAvgPool2d(int(image_shape[0]*0.9)),  torch.nn.LeakyReLU(), 
                                      torch.nn.Conv2d(128,128,kernel_size=2), torch.nn.AdaptiveAvgPool2d(min_encoding_size)
                                    ]) #B,C,H,W
        self.add_module("conv_sequential", convs)
    def forward(self, images: "B, C, H, W"):
        x = images #to be converted to normalized [-1 to 1] distance map
        x = self.conv_sequential(x) 
        return x
        
class DecoderConv(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        image_shape = kwargs.get("image_shape") #H,W
        picture_dim = kwargs.get("picture_dim")
        convs = torch.nn.Sequential(*[torch.nn.ConvTranspose2d(128,128,kernel_size=2), torch.nn.Upsample(int(image_shape[0]*0.9), mode='nearest'), torch.nn.LeakyReLU(), 
                                      torch.nn.ConvTranspose2d(128,64,kernel_size=2), torch.nn.Upsample(int(image_shape[0]*0.95), mode='nearest'),  torch.nn.LeakyReLU(), 
                                      torch.nn.ConvTranspose2d(64,picture_dim,kernel_size=3), torch.nn.Upsample(image_shape[0], mode='nearest'), 
                                    ]) #B,C,H,W
        self.add_module("conv_sequential", convs)
    def forward(self, images: "B, C, H, W"):
        x = images
        x = self.conv_sequential(x) 
        return x

class PatchDiscriminator(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        image_shape = kwargs.get("image_shape") #H,W
        picture_dim = kwargs.get("picture_dim") 
        convs = torch.nn.Sequential(*[torch.nn.Conv2d(picture_dim,32,kernel_size=5,groups=1,stride=1,padding="same"), torch.nn.Conv2d(32,16,kernel_size=3,groups=1,stride=1), 
                                      torch.nn.Conv2d(16,1,kernel_size=1,groups=1,stride=1), torch.nn.Sigmoid()
                                    ]) #B,C,H,W -> NAN!!! <<< Instance, Depth, Instance to compare with (B,1,H,W)
        self.add_module("conv_sequential", convs)
    def forward(self, images: "B, C, H, W"):
        x = images
        x = self.conv_sequential(x) 
        return x

class Discriminator(torch.nn.Module):
    def __init__(self, **kwargs):
        super(Discriminator, self).__init__()
        picture_dim = kwargs.get("picture_dim") 
        image_shape = kwargs.get("image_shape") #H,W
        self.model = torch.nn.Sequential (
                    torch.nn.Conv2d(picture_dim, 32, 4, 1, 0),
                    torch.nn.BatchNorm2d(32),
                    torch.nn.ReLU(True),
                    torch.nn.AdaptiveAvgPool2d(image_shape[0]//2),
                    torch.nn.Conv2d(32, 32, 3, 2, 1),
                    torch.nn.BatchNorm2d(32),
                    torch.nn.ReLU(True),
                    torch.nn.AdaptiveAvgPool2d(image_shape[0]//4),
                    torch.nn.Conv2d(32, 32, 3, 2, 1),
                    torch.nn.BatchNorm2d(32),
                    torch.nn.ReLU(True),
                    torch.nn.AdaptiveAvgPool2d(1),
                )
        self.out = torch.nn.Linear(32, 1)
    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 32)
        x = self.out(x)
        return torch.sigmoid(x)

def distance_normalizer(reference: torch.Tensor, trajectory: torch.Tensor):
    bbox = torch.cat((reference.min(dim=0).values.view(-1,1), reference.max(dim=0).values.view(-1,1)), dim=0)
    max_dist = torch.sqrt(torch.sum(((bbox[1,:] - bbox[0,:])**2))) + 30
    variance = trajectory.var()
    return [max_dist.item(), variance]

class VQVAE(torch.nn.Module):
    """GANs
    https://github.com/eriklindernoren/PyTorch-GAN/tree/master/implementations
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.n_e = n_e = kwargs.get("n_e")
        self.min_encoding_size = min_encoding_size = kwargs.get("min_encoding_size")
        self.image_shape = image_shape = kwargs.get("image_shape")
        self.picture_dim = picture_dim = kwargs.get("picture_dim")
        self.dm = DistanceMapping()
        self.ec = EncoderConv(image_shape=image_shape, min_encoding_size=min_encoding_size,picture_dim=picture_dim)
        self.dc = DecoderConv(image_shape=image_shape,picture_dim=picture_dim)
        self.vq = VectorQuantizer(n_e, 128, 1)
    def forward(self, ref: """Protein pdb of B,L,3""", inputs: """Protein trajectories of B,L,3""", normalizing_factor: List[int]):
        dist_map = self.dm(inputs)[:,None,...] / normalizing_factor[0] -0.5 #B1HW with -0.5 to 0.5 range.
        rbf = dist_map #Not real RBF
        encC = self.ec(rbf) #Range of 0-1 input
        z_q, vqloss, (perplexity, min_encodings, min_encoding_indices) = self.vq(encC)
        decC = self.dc(z_q) #Range of 0-1 output
        convloss = torch.nn.MSELoss()(rbf, decC) / normalizing_factor[1] #Data variance

        total_loss = vqloss + convloss
        print(f"vqloss {vqloss.item()} convloss {convloss.item()}")
        return total_loss

class VQVAE_GT(torch.nn.Module):
    """GANs
    https://github.com/eriklindernoren/PyTorch-GAN/tree/master/implementations
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.n_e = n_e = kwargs.get("n_e")
        self.min_encoding_size = min_encoding_size = kwargs.get("min_encoding_size")
        self.image_shape = image_shape = kwargs.get("image_shape")
        self.picture_dim = picture_dim = kwargs.get("picture_dim")
        self.prev_model_file = prev_model_file = kwargs.get("prev_model_file")
        self.transfer_model = self._freeze_and_load(self.prev_model_file)
        self.gs2 = RBFLayer_Physnet(dfilter=picture_dim, cutoff=5)
        self.gt = GraphTransformer(dim=3, depth=4, edge_dim=picture_dim)
    def _freeze_and_load(self, prev_model_file):
        _prev_model_ckpt = torch.load(prev_model_file, map_location=device)
        _prev_model = VQVAE(n_e=self.n_e, min_encoding_size=self.min_encoding_size, image_shape=self.image_shape, picture_dim=1) #picture dim is different!
        _prev_model.load_state_dict(_prev_model_ckpt)
        for p in _prev_model.parameters():
            p.requires_grad = False
        print(fmt.red(f"Model successfully frozen..."))
        return _prev_model.dm
    def forward(self, ref: """Protein pdb of B,L,3""", inputs: """Protein trajectories of B,L,3""", normalizing_factor: List[int]):
        dist_map = self.transfer_model(inputs)[:,None,...] / normalizing_factor[0] -0.5 #Only DM module!
        #encC = self.transfer_model.ec(rbf) 
        #z_q, vqloss, (perplexity, min_encodings, min_encoding_indices) = self.transfer_model.vq(encC)
        #decC = self.transfer_model.dc(z_q) 

        rbf = self.gs2(dist_map) #Using teacher forcing!
        with torch.no_grad():
            rbf_det = rbf.clone()
            #decC_det = decC.clone()
            rbf_det.set_(rbf)
            #decC_det.set_(decC)

        nodes, edges = self.gt(ref, rbf_det) #Teacher forcing is RBF_det, student forcing is decC_det <comes from 1. VAE latent 2. Codebook recon 3. z_q and decC --pass to GS module-- 4. GT module>
        gtloss = torch.nn.MSELoss()(inputs, nodes)

        dist_map_fake = self.transfer_model(nodes)[:,None,...] / normalizing_factor[0] -0.5
        dist_map_loss = torch.nn.MSELoss()(dist_map, dist_map_fake)

        total_loss = gtloss + dist_map_loss
        print(f"gtloss {gtloss.item()} dist_map_loss {dist_map_loss.item()}")
        return total_loss

class VQVAE_GAN(torch.nn.Module):
    """GANs
    https://github.com/eriklindernoren/PyTorch-GAN/tree/master/implementations
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.n_e = n_e = kwargs.get("n_e")
        self.min_encoding_size = min_encoding_size = kwargs.get("min_encoding_size")
        self.image_shape = image_shape = kwargs.get("image_shape")
        self.picture_dim = picture_dim = kwargs.get("picture_dim")
        self.prev_model_file = prev_model_file = kwargs.get("prev_model_file")
        self.transfer_model = self._freeze_and_load(self.prev_model_file)
        self.pd = Discriminator(picture_dim=picture_dim, image_shape=image_shape) #normal discriminator
    def _freeze_and_load(self, prev_model_file):
        _prev_model_ckpt = torch.load(prev_model_file, map_location=device)
        _prev_model = VQVAE(n_e=self.n_e, min_encoding_size=self.min_encoding_size, image_shape=self.image_shape, picture_dim=self.picture_dim)
        _prev_model.load_state_dict(_prev_model_ckpt)
        for p in _prev_model.parameters():
            p.requires_grad = False
        print(fmt.red(f"Model successfully frozen..."))
        return _prev_model
    def forward(self, ref: """Protein pdb of B,L,3""", inputs: """Protein trajectories of B,L,3""", normalizing_factor: List[int]):
        dist_map = self.transfer_model.dm(inputs)[:,None,...] / normalizing_factor[0] -0.5
        rbf = dist_map 
        encC = self.transfer_model.ec(rbf) 
        z_q, vqloss, (perplexity, min_encodings, min_encoding_indices) = self.transfer_model.vq(encC)
        decC = self.transfer_model.dc(z_q) 

        with torch.no_grad():
            rbf_det = rbf.clone()
            decC_det = decC.clone()
            rbf_det.set_(rbf)
            decC_det.set_(decC)

        adversarial_loss_fn = torch.nn.functional.binary_cross_entropy
        self.register_buffer("valid", torch.ones(rbf.size(0), 1).float(), persistent=False)
        self.register_buffer("fake", torch.zeros(rbf.size(0), 1).float(), persistent=False)
        real_loss = adversarial_loss_fn(self.pd(rbf_det).view(-1,1), self.valid.to(device))
        fake_loss = adversarial_loss_fn(self.pd(decC_det).view(-1,1), self.fake.to(device))
        d_loss = ((real_loss + fake_loss) / 2).mean() #With BCE
        #d_loss = -self.pd(rbf_det).mean(dim=(1,2,3)).view(-1,1).mean() + self.pd(decC_det).mean(dim=(1,2,3)).view(-1,1).mean() #Without

        alphas = torch.rand_like(rbf_det)
        interpolated = alphas * rbf_det.data + (1 - alphas) * decC_det.data
        interpolated.requires_grad_()
        interpolated = interpolated.to(device)
        prob_interpolated = self.pd(interpolated)
        gradients = torch.autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(prob_interpolated.size()).to(device),
                               create_graph=True, retain_graph=True)[0]
        gradients = gradients.contiguous().view(rbf_det.size(0), -1) #B by -1
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
        gp_loss = ((gradients_norm - 1) ** 2).mean() #GP loss

        total_loss = d_loss + gp_loss
        print(f"d_loss {d_loss.item()} gp_loss {gp_loss.item()}")
        return total_loss

class VQVAE_VAE(torch.nn.Module):
    """GANs
    https://github.com/eriklindernoren/PyTorch-GAN/tree/master/implementations
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.n_e = n_e = kwargs.get("n_e")
        self.min_encoding_size = min_encoding_size = kwargs.get("min_encoding_size")
        self.image_shape = image_shape = kwargs.get("image_shape")
        self.picture_dim = picture_dim = kwargs.get("picture_dim")
        self.prev_model_file = prev_model_file = kwargs.get("prev_model_file")
        self.transfer_model = self._freeze_and_load(self.prev_model_file)
        self.vae = VAE(n_e=n_e, image_shape=min_encoding_size)
    def _freeze_and_load(self, prev_model_file):
        _prev_model_ckpt = torch.load(prev_model_file, map_location=device)
        _prev_model = VQVAE(n_e=self.n_e, min_encoding_size=self.min_encoding_size, image_shape=self.image_shape, picture_dim=self.picture_dim)
        _prev_model.load_state_dict(_prev_model_ckpt)
        for p in _prev_model.parameters():
            p.requires_grad = False
        print(fmt.red(f"Model successfully frozen..."))
        return _prev_model
    def forward(self, ref: """Protein pdb of B,L,3""", inputs: """Protein trajectories of B,L,3""", normalizing_factor: List[int]):
        dist_map = self.transfer_model.dm(inputs)[:,None,...] / normalizing_factor[0] -0.5
        rbf = dist_map 
        encC = self.transfer_model.ec(rbf) 
        z_q, vqloss, (perplexity, min_encodings, min_encoding_indices) = self.transfer_model.vq(encC)
        decC = self.transfer_model.dc(z_q) 

        recon_scaled, z, mu, logstd, recon = self.vae(min_encoding_indices)
        sizes = min_encoding_indices.size()
        tmp = min_encoding_indices.view(sizes[0],1,sizes[1],sizes[2])
        input_scaled = self.vae.enc.normalize(tmp)
        input_scaled.requires_grad = False
        vaeloss = self.vae.losses(input_scaled, recon_scaled, z, mu, logstd)

        total_loss = vaeloss
        print(f"vaeloss {vaeloss.item()}")
        return total_loss


###########################
#######DATALOADING#########
###########################
torch.manual_seed(42)

from MDAnalysis.tests.datafiles import CRD, PSF, DCD, DCD2
from MDAnalysis.analysis import align
from MDAnalysis.coordinates.memory import MemoryReader
from MDAnalysis.analysis.base import AnalysisFromFunction

adk = mda.Universe(CRD)
adk_open = mda.Universe(CRD, DCD2)
adk_closed = mda.Universe(PSF, DCD)
at = align.AlignTraj(adk_closed,  # trajectory to align
                adk,  # reference
                select='name CA',  # selection of atoms to align
                filename='alignedc.pdb',  # file to write the trajectory to              
               )
at.run()
at = align.AlignTraj(adk_open,  # trajectory to align
                adk,  # reference
                select='name CA',  # selection of atoms to align
                filename='alignedo.pdb',  # file to write the trajectory to                          
               )
at.run()

atom_selection = "name CA"
copen = AnalysisFromFunction(lambda ag: ag.positions.copy(),
                                   adk_open.atoms.select_atoms(f"{atom_selection}")).run().results['timeseries']
cclosed = AnalysisFromFunction(lambda ag: ag.positions.copy(),
                                   adk_closed.atoms.select_atoms(f"{atom_selection}")).run().results['timeseries']
ref = torch.from_numpy(adk.atoms.select_atoms(f"{atom_selection}").positions)[None,:]
traj = torch.from_numpy(copen) #Test
trajv = torch.from_numpy(cclosed) #Validation
image_size = ref.size(1)
const = distance_normalizer(ref, traj) #Normalizing dist

###########################
##########TEST#############
###########################
ref, traj, trajv = ref.to(device), traj.to(device), trajv.to(device)
T = torch.cat((traj, trajv),dim=0).to(torch.device("cpu"))

"""
##VQVAE Only
device = torch.device("cpu")
CurrentModel = VQVAE
prev_model_file = None #"ckpt_vqvae.pt"
current_model_file = "ckpt_vqvae.pt"
min_encoding_size=int(image_size*0.65)

model = CurrentModel(n_e=int(min_encoding_size), min_encoding_size=min_encoding_size, image_shape=[image_size]*2, sequence_length=image_size, picture_dim=1, prev_model_file=prev_model_file)
#device = torch.device("cpu")
ckpt = torch.load(current_model_file, map_location=device)
model.load_state_dict(ckpt)
model.to(device)
model.eval()

dist_map = model.dm(T)[:,None,...] / const[0] -0.5 #B1HW with 0-1 range.
rbf = dist_map #Not real RBF
encC = model.ec(rbf) #Range of 0-1 input
z_q, vqloss, (perplexity, min_encodings, min_encoding_indices) = model.vq(encC)
decC = model.dc(z_q) #Range of 0-1 output
import matplotlib.pyplot as plt
fig,ax=plt.subplots(3,1); ax[0].imshow(min_encoding_indices.detach().cpu().numpy()[5]); ax[1].imshow(rbf.detach().cpu().numpy()[5,0]); ax[2].imshow(decC.detach().cpu().numpy()[5,0]); plt.savefig("t1.png")
"""


"""
##VQVAE-GT
T = T.to(device)[:5]
CurrentModel = VQVAE_GT
prev_model_file = "ckpt_vqvae.pt" #"ckpt_vqvae.pt"
current_model_file = "ckpt_vqvae_gt.pt"
min_encoding_size=int(image_size*0.65)

model = CurrentModel(n_e=int(min_encoding_size), min_encoding_size=min_encoding_size, image_shape=[image_size]*2, sequence_length=image_size, picture_dim=64, prev_model_file=prev_model_file)
#device = torch.device("cpu")
ckpt = torch.load(current_model_file, map_location=device)
model.load_state_dict(ckpt)
model.to(device)
model.eval()

refmean = ref[0].mean(dim=0, keepdim=True)
ref = ref - refmean
T = T - refmean[:,None,:]
ref = ref / 10.
T = T / 10.

dist_map = model.transfer_model(T)[:,None,...] / const[0] -0.5 #B1HW with 0-1 range.
rbf = model.gs2(dist_map) #Not real RBF
#encC = model.transfer_model.ec(rbf) #Range of 0-1 input
#z_q, vqloss, (perplexity, min_encodings, min_encoding_indices) = model.transfer_model.vq(encC)
#decC = model.transfer_model.dc(z_q) #Range of 0-1 output

rbf_det = rbf.clone()
rbf_det.set_(rbf).detach()
#decC_det = decC.clone()
#decC_det.set_(decC).detach()
nodes, edges = model.gt(ref.expand_as(T), rbf_det) #Nodes is to T AS decC_det is to rbf_det

nodes = nodes * 10.
nodes = nodes + refmean[:,None,:]
fake_closed = mda.Universe(CRD)
fake_closed_ca = fake_closed.atoms.select_atoms(atom_selection)
fake_closed_ca.atoms.positions = nodes[4].detach().cpu().numpy()
fake_closed_ca.write("fake.pdb")
"""

"""
##VQVAE-VAE
CurrentModel = VQVAE_VAE
prev_model_file = "ckpt_vqvae.pt" #"ckpt_vqvae.pt"
current_model_file = "ckpt_vqvae_vae_hook.pt"
min_encoding_size=int(image_size*0.65)

model = CurrentModel(n_e=int(min_encoding_size), min_encoding_size=min_encoding_size, image_shape=[image_size]*2, sequence_length=image_size, picture_dim=1, prev_model_file=prev_model_file)
device = torch.device("cpu")
ckpt = torch.load(current_model_file, map_location=device)
model.load_state_dict(ckpt)
model.to(device)
model.eval()

T_ = T.requires_grad_()
dist_map = model.transfer_model.dm(T_)[:,None,...] / const[0] -0.5 #B1HW with 0-1 range.
rbf = dist_map #Not real RBF
encC = model.transfer_model.ec(rbf) #Range of 0-1 input
z_q, vqloss, (perplexity, min_encodings, min_encoding_indices) = model.transfer_model.vq(encC)
#decC = model.transfer_model.dc(z_q) #Range of 0-1 output
recon_scaled, z, mu, logstd, recon = model.vae(min_encoding_indices)

mei = min_encoding_indices.float()[:3].requires_grad_() #Do not select indices AFTER giving gradient!
z_, mu_, logstd_ = model.vae.enc(mei)
z1 = mu_
lr = 1e4
fig,ax=plt.subplots(3,2);
ax[0,0].imshow(mei.detach().cpu().numpy()[0])
ax[0,1].scatter(*mu_.detach().cpu().numpy().T)
grads = torch.zeros_like(mei.data)
for step in tqdm.tqdm(range(1000)):
    z_, mu_, logstd_ = model.vae.enc(mei); mu_.backward(torch.ones_like(mu_)); mei.data = mei.data - lr * mei.grad.data; grads += lr * mei.grad.data
z2 = mu_
ax[1,0].imshow(mei.detach().cpu().numpy()[0])
ax[1,1].scatter(*mu_.detach().cpu().numpy().T)
ax[2,0].imshow(mei.detach().cpu().numpy()[0]*0. + min_encoding_indices.float().detach().cpu().numpy()[98]*1.)
ax[2,1].imshow(grads.detach().cpu().numpy()[0]*1e2, cmap=plt.cm.get_cmap("jet"))
plt.savefig("vae_backpropZ_codebook.png")


#import matplotlib.pyplot as plt
#plt.scatter(*mu.detach().cpu().numpy()[:traj.size(0)].T, c=np.arange(len(mu[:traj.size(0)])), edgecolor='r', linewidth=0.5); 
#plt.scatter(*mu.detach().cpu().numpy()[traj.size(0):].T, c=np.arange(len(mu[traj.size(0):])), edgecolor='b', linewidth=0.5); 
#plt.scatter(*mu.detach().cpu().numpy()[:traj.size(0)].T, c='r'); 
#plt.scatter(*mu.detach().cpu().numpy()[traj.size(0):].T, c='b'); 
#plt.show()

#fig,ax=plt.subplots(2,1); ax[0].imshow(min_encoding_indices.detach().cpu().numpy()[15]); ax[1].imshow(recon.detach().cpu().numpy().astype(np.int32)[15]);  plt.show()

#model.transfer_model.get_codebook_entry()
"""



##VQVAE-GAN
CurrentModel = VQVAE_GAN
prev_model_file = "ckpt_vqvae.pt" #"ckpt_vqvae.pt"
current_model_file = "ckpt_vqvae_gan.pt"
min_encoding_size=int(image_size*0.65)

model = CurrentModel(n_e=int(min_encoding_size), min_encoding_size=min_encoding_size, image_shape=[image_size]*2, sequence_length=image_size, picture_dim=1, prev_model_file=prev_model_file)
device = torch.device("cpu")
ckpt = torch.load(current_model_file, map_location=device)
model.load_state_dict(ckpt)
model.to(device)
model.eval()

feat = dict()
grad_feat = dict()
def hookf(m, i, o):
    feat["pd_9"] = o
def hookb(m, i, o):
    grad_feat["pd_9"] = o
model.pd.model[8].register_forward_hook(hookf)
model.pd.model[8].register_backward_hook(hookb)

dist_map = model.transfer_model.dm(T)[:,None,...] / const[0] -0.5 #B1HW with 0-1 range.
rbf = dist_map #Not real RBF
encC = model.transfer_model.ec(rbf) #Range of 0-1 input
z_q, vqloss, (perplexity, min_encodings, min_encoding_indices) = model.transfer_model.vq(encC)
decC = model.transfer_model.dc(z_q) #Range of 0-1 output

rbf_det = rbf.clone()
rbf_det.set_(rbf).detach()
decC_det = decC.clone()
decC_det.set_(decC).detach()

def gradcam(plusplus=False, idx=None):
    fig, ax = plt.subplots(2,2)
    if idx == None:
        idx = 0

    losses = model.pd(rbf_det).view(-1,1)
    losses.backward(gradient=torch.ones_like(losses)) #Just following computational graph...
    activations = feat["pd_9"]
    grads = grad_feat["pd_9"][0]
    if plusplus:
        grads_power_2 = grads**2
        grads_power_3 = grads_power_2 * grads
        sum_activations = torch.sum(activations, dim=(2, 3))
        eps = torch.finfo().eps
        aij = grads_power_2 / (2 * grads_power_2 +
                           sum_activations[:, :, None, None] * grads_power_3 + eps)
        aij = torch.where(grads != 0., aij, torch.tensor(0.))
        #weights = torch.amax(grads, dim=0) * aij #Use relu
        weights = torch.nn.functional.relu(grads) * aij
        weights = torch.sum(weights, axis=(2, 3)) 
    else:
        weights = grads.sum(dim=(2,3)) / (activations.size(2)*activations.size(3))
    weighted_activations = weights[:, :, None, None] * activations
    cam = torch.nn.functional.relu(weighted_activations.sum(axis=1))
    resized_cam = cv2.resize(cam[idx].detach().cpu().numpy(), (rbf_det.size(2), rbf_det.size(3)))
    resized_cam -= np.min(resized_cam)
    resized_cam /= np.max(resized_cam)
    resized_cam_jet = plt.cm.jet(resized_cam)[...,:3]
    ax[0,0].imshow(rbf.detach().cpu().numpy()[idx,0]); ax[0,0].imshow(resized_cam_jet, alpha=0.8); 
    ax[1,0].imshow(rbf.detach().cpu().numpy()[idx,0]);

    losses = model.pd(decC_det).view(-1,1)
    losses.backward(gradient=torch.ones_like(losses)) #Just following computational graph...
    activations = feat["pd_9"]
    grads = grad_feat["pd_9"][0]
    if plusplus:
        grads_power_2 = grads**2
        grads_power_3 = grads_power_2 * grads
        sum_activations = torch.sum(activations, dim=(2, 3))
        eps = torch.finfo().eps
        aij = grads_power_2 / (2 * grads_power_2 +
                           sum_activations[:, :, None, None] * grads_power_3 + eps)
        aij = torch.where(grads != 0., aij, torch.tensor(0.))
        #weights = torch.amax(grads, dim=0) * aij #Use relu
        weights = torch.nn.functional.relu(grads) * aij
        weights = torch.sum(weights, axis=(2, 3)) 
    else:
        weights = grads.sum(dim=(2,3)) / (activations.size(2)*activations.size(3))
    weights = grads.sum(dim=(2,3)) / (activations.size(2)*activations.size(3))
    weighted_activations = weights[:, :, None, None] * activations
    cam = torch.nn.functional.relu(weighted_activations.sum(axis=1))
    resized_cam = cv2.resize(cam[idx].detach().cpu().numpy(), (rbf_det.size(2), rbf_det.size(3)))
    resized_cam -= np.min(resized_cam)
    resized_cam /= np.max(resized_cam)
    resized_cam_jet = plt.cm.jet(resized_cam)[...,:3]
    ax[0,1].imshow(decC.detach().cpu().numpy()[idx,0]); ax[0,1].imshow(resized_cam_jet, alpha=0.8); 
    ax[1,1].imshow(decC.detach().cpu().numpy()[idx,0]);
    plt.show()
gradcam(True, idx=50)

def saliency(idx = None):
    fig, ax = plt.subplots(2,2)
    if idx == None:
        idx = 0
    rbf_det.requires_grad_()
    losses = model.pd(rbf_det).view(-1,1)
    losses.backward(gradient=torch.ones_like(losses)) #Just following computational graph...
    ax[0,0].imshow(rbf_det.grad.data.abs().amax(dim=1).detach().cpu().numpy()[idx], cmap=plt.cm.get_cmap("gray"));
    ax[1,0].imshow(rbf_det.detach().cpu().numpy()[idx,0]);
    decC_det.requires_grad_()
    losses = model.pd(decC_det).view(-1,1)
    losses.backward(gradient=torch.ones_like(losses)) #Just following computational graph...
    ax[0,1].imshow(decC_det.grad.data.abs().amax(dim=1).detach().cpu().numpy()[idx], cmap=plt.cm.get_cmap("gray")); 
    ax[1,1].imshow(decC_det.detach().cpu().numpy()[idx,0]);
    plt.show()
saliency(idx=50)


