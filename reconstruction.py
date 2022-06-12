import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
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

warnings.simplefilter("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_printoptions(precision=4)

"""
class Encoder(torch.nn.Module):
    def __init__(self, dim=[128,64,4], **kwargs):
        super().__init__()
        self.dim = dim
        self.voc_size = kwargs.get("n_e")
        convs = torch.nn.Sequential(*[ 
                                      torch.nn.Conv2d(1,64,kernel_size=3), torch.nn.AdaptiveAvgPool2d(64), torch.nn.LeakyReLU(), torch.nn.BatchNorm2d(64),
                                      torch.nn.Conv2d(64,128,kernel_size=2), torch.nn.AdaptiveAvgPool2d(32),  torch.nn.LeakyReLU(), torch.nn.BatchNorm2d(128),
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
        convs = torch.nn.Sequential(*[torch.nn.ConvTranspose2d(128,128, kernel_size=2), torch.nn.Upsample(32, mode='nearest'), torch.nn.LeakyReLU(), torch.nn.BatchNorm2d(128),
                              torch.nn.ConvTranspose2d(128,64, kernel_size=2), torch.nn.Upsample(64, mode='nearest'),  torch.nn.LeakyReLU(), torch.nn.BatchNorm2d(64),
                              torch.nn.ConvTranspose2d(64,1,kernel_size=3), torch.nn.Upsample(image_shape, mode='nearest'), torch.nn.Sigmoid()
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
        image = image * (self.voc_size - 1)
        return image    
# dec = Decoder(n_e=100, image_shape=min_encoding_indices.size(1))
# x, x_ = dec(z)

class VAE(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        voc_size = kwargs.get("n_e")
        image_shape = kwargs.get("image_shape")
        self.enc = Encoder(n_e=voc_size)
        self.dec = Decoder(n_e=voc_size, image_shape=image_shape)
        # self.apply(self._init_weights)
    def forward(self, inputs):
        x = inputs
        en, mu, logstd = self.enc(x)
        x, image = self.dec(en)
        return x, en, mu, logstd, image
    def losses(self, inputs, recon, z, mu, logstd):
        #bce = torch.nn.BCELoss()(recon, inputs)
        bce = torch.nn.BCELoss(reduction="none")(recon, inputs).mean(dim=(1,2,3)).mean()
        #kl = torch.distributions.Normal(mu, logstd.exp()).log_prob(z) - torch.distributions.Normal(0., 1.).log_prob(z)
        kl = torch.mean(-0.5 * torch.sum(1 + logstd - mu ** 2 - logstd.exp(), dim = 1), dim = 0)
        return bce + kl
    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear) and hasattr(m, 'weight') and m.weight is not None:
            torch.nn.init.uniform_(m.weight,0,1)
        if isinstance(m, torch.nn.Linear) and hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0.)
        if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)) and hasattr(m, 'weight') and m.weight is not None:
            torch.nn.init.uniform_(m.weight)
        if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)) and hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0.)
"""

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
        return x + torch.log(-torch.expm1(-x))

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


class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost
        
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()
        
        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)
            
            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon) * n)
            
            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)
            
            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss
        
        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings

"""
class VectorQuantizer(torch.nn.Module):
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
"""

"""
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
        with_feedforwards = False,
        norm_edges = False,
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
        edges = self.norm_edges(edges)
        for attn_block, ff_block in self.layers:
            attn, attn_residual = attn_block
            nodes = attn_residual(attn(nodes, edges, mask = mask), nodes)
            if exists(ff_block):
                ff, ff_residual = ff_block
                nodes = ff_residual(ff(nodes), nodes)
        return nodes, edges
"""

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


class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False)
        )
    
    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                             for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)

class EncoderConv(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Encoder, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens//2,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_2 = nn.Conv2d(in_channels=num_hiddens//2,
                                 out_channels=num_hiddens,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_3 = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = F.relu(x)
        
        x = self._conv_2(x)
        x = F.relu(x)
        
        x = self._conv_3(x)
        return self._residual_stack(x)

class DecoderConv(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Decoder, self).__init__()
        
        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens,
                                 kernel_size=3, 
                                 stride=1, padding=1)
        
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)
        
        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens, 
                                                out_channels=num_hiddens//2,
                                                kernel_size=4, 
                                                stride=2, padding=1)
        
        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens//2, 
                                                out_channels=3,
                                                kernel_size=4, 
                                                stride=2, padding=1)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        
        x = self._residual_stack(x)
        
        x = self._conv_trans_1(x)
        x = F.relu(x)
        
        return self._conv_trans_2(x)


"""
class EncoderConv(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        image_shape = kwargs.get("image_shape") #H,W
        min_encoding_size = kwargs.get("min_encoding_size")
        picture_dim = kwargs.get("picture_dim") 
        convs = torch.nn.Sequential(*[ 
                                      torch.nn.Conv2d(picture_dim,64,kernel_size=3), torch.nn.AdaptiveAvgPool2d(int(image_shape[0]*0.95)), torch.nn.LeakyReLU(), torch.nn.BatchNorm2d(64),
                                      torch.nn.Conv2d(64,128,kernel_size=2), torch.nn.AdaptiveAvgPool2d(int(image_shape[0]*0.9)),  torch.nn.LeakyReLU(), torch.nn.BatchNorm2d(128),
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
        convs = torch.nn.Sequential(*[torch.nn.ConvTranspose2d(128,128,kernel_size=2), torch.nn.Upsample(int(image_shape[0]*0.9), mode='nearest'), torch.nn.LeakyReLU(), torch.nn.BatchNorm2d(128),
                                      torch.nn.ConvTranspose2d(128,64,kernel_size=2), torch.nn.Upsample(int(image_shape[0]*0.95), mode='nearest'),  torch.nn.LeakyReLU(), torch.nn.BatchNorm2d(64),
                                      torch.nn.ConvTranspose2d(64,picture_dim,kernel_size=3), torch.nn.Upsample(image_shape[0], mode='nearest'), 
                                    ]) #B,C,H,W
        self.add_module("conv_sequential", convs)
    def forward(self, images: "B, C, H, W"):
        x = images
        x = self.conv_sequential(x) 
        return x
"""

class PatchDiscriminator(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        image_shape = kwargs.get("image_shape") #H,W
        picture_dim = kwargs.get("picture_dim") 
        convs = torch.nn.Sequential(*[torch.nn.Conv2d(picture_dim,32,kernel_size=1,groups=1,stride=1), torch.nn.Conv2d(32,16,kernel_size=1,groups=16,stride=1), 
                                      torch.nn.Conv2d(16,1,kernel_size=1,groups=1,stride=1), torch.nn.Sigmoid()
                                    ]) #B,C,H,W ->Instance, Depth, Instance to compare with (B,1,H,W)
        self.add_module("conv_sequential", convs)
    def forward(self, images: "B, C, H, W"):
        x = images
        x = self.conv_sequential(x) 
        return x

class DistanceNormalizer(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        convs = torch.nn.Sequential(*[torch.nn.Conv2d(picture_dim,32,kernel_size=1,groups=1,stride=1), torch.nn.Conv2d(32,16,kernel_size=1,groups=16,stride=1), 
                                      torch.nn.Conv2d(16,1,kernel_size=1,groups=1,stride=1), torch.nn.Sigmoid()
                                    ]) #B,C,H,W ->Instance, Depth, Instance to compare with (B,1,H,W)
        self.add_module("conv_sequential", convs)
    def forward(self, images: "B, C, H, W"):
        x = images
        x = self.conv_sequential(x) 
        return x

class Monster(torch.nn.Module):
    """GANs
    https://github.com/eriklindernoren/PyTorch-GAN/tree/master/implementations
    """
    def __init__(self, **kwargs):
        super().__init__()
        num_hiddens = 128
        num_residual_hiddens = 32
        num_residual_layers = 2
        embedding_dim = 64
        num_embeddings = 512
        commitment_cost = 0.25
        decay = 0.99

        n_e = kwargs.get("n_e")
        min_encoding_size = kwargs.get("min_encoding_size")
        image_shape = kwargs.get("image_shape")
        self.dm = DistanceMapping()
        #picture_dim = 3
        #self.gs = GaussianSmearing(n_gaussians=picture_dim)
        #self.gs2 = RBFLayer_Physnet(dfilter=picture_dim, cutoff=5)
        self.ec = EncoderConv(3, num_hiddens,
                                num_residual_layers, 
                                num_residual_hiddens)
        self.dc = DecoderConv(embedding_dim,
                                num_hiddens, 
                                num_residual_layers, 
                                num_residual_hiddens)
        self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens, 
                                      out_channels=embedding_dim,
                                      kernel_size=1, 
                                      stride=1)
        self.vq = ectorQuantizerEMA(num_embeddings, embedding_dim, 
                                              commitment_cost, decay)
        self.pd = PatchDiscriminator(picture_dim=picture_dim)
        self.gt = GraphTransformer(3,2,edge_dim=picture_dim)
        self.vae = VAE(n_e=n_e, image_shape=min_encoding_size)
    def forward_intermediates(self, ref: """Protein pdb of B,L,3""", inputs: """Protein trajectories of B,L,3"""):
        dist_map = self.dm(inputs)[:,None,...] #B1HW
        rbf = self.gs2(dist_map) #BCHW
        encC = self.ec(rbf)
        encC = self._pre_vq_conv(encC)
        loss, quantized, perplexity, _ = self.vq(encC)

        decC = self.dc(z_q)
        convloss = torch.nn.MSELoss()(rbf, decC)

        adversarial_loss_fn = torch.nn.functional.binary_cross_entropy
        #self.register_buffer("valid", torch.ones(rbf.size(0), 1).float(), persistent=False)
        #self.register_buffer("fake", torch.zeros(rbf.size(0), 1).float(), persistent=False)
        # g_loss = adversarial_loss_fn(self.pd(decC).mean(dim=(1,2,3), keepdim=True), valid)
        #print(self.pd(rbf.detach()))
        with torch.no_grad():
            rbf_det = rbf.clone()
            decC_det = decC.clone()
            rbf_det.set_(rbf)
            decC_det.set_(decC)
        #print(self.valid, rbf_det)
        #real_loss = adversarial_loss_fn(self.pd(rbf_det).mean(dim=(1,2,3)).view(-1,1), self.valid.to(device))
        #fake_loss = adversarial_loss_fn(self.pd(decC_det).mean(dim=(1,2,3)).view(-1,1), self.fake.to(device))
        #d_loss = ((real_loss + fake_loss) / 2).mean() #With BCE
        d_loss = -self.pd(rbf_det).mean(dim=(1,2,3)).view(-1,1).mean() + self.pd(decC_det).mean(dim=(1,2,3)).view(-1,1).mean() #Without BCE
        alphas = torch.rand_like(rbf_det)
        interpolated = alphas * rbf_det.data + (1 - alphas) * decC_det.data
        interpolated.requires_grad_()
        interpolated = interpolated.to(device)
        prob_interpolated = self.pd(interpolated)
        gradients = torch.autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(prob_interpolated.size()).to(device),
                               create_graph=True, retain_graph=True)[0]
        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.contiguous().view(rbf_det.size(0), -1) #B by -1
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
        gp_loss = ((gradients_norm - 1) ** 2).mean() #GP loss

        recon_scaled, z, mu, logstd, recon = self.vae(min_encoding_indices)
        sizes = min_encoding_indices.size()
        tmp = min_encoding_indices.view(sizes[0],1,sizes[1],sizes[2])
        input_scaled = self.vae.enc.normalize(tmp)
        input_scaled.requires_grad = False
        vaeloss = self.vae.losses(input_scaled, recon_scaled, z, mu, logstd)
        nodes, edges = self.gt(ref, decC)
        gtloss = torch.nn.MSELoss()(inputs, nodes)

        total_loss = vqloss + convloss + d_loss + gp_loss + vaeloss #+ gtloss
        print(f"vqloss {vqloss.item()} convloss {convloss.item()} d_loss {d_loss.item()} gp_loss {gp_loss.item()} vaeloss {vaeloss.item()} gtloss {gtloss.item()}")
        return total_loss


#Monster
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
#merged3 = mda.Merge(adk_closed.atoms, adk_open.atoms)

atom_selection = "name CA"
copen = AnalysisFromFunction(lambda ag: ag.positions.copy(),
                                   adk_open.atoms.select_atoms(f"{atom_selection}")).run().results['timeseries']
cclosed = AnalysisFromFunction(lambda ag: ag.positions.copy(),
                                   adk_closed.atoms.select_atoms(f"{atom_selection}")).run().results['timeseries']
ref = torch.from_numpy(adk.atoms.select_atoms(f"{atom_selection}").positions)[None,:]
traj = torch.from_numpy(copen)
trajv = torch.from_numpy(cclosed)
image_size = ref.size(1)

#print(adk.trajectory.n_frames,ref.size(), traj.size(), trajv.size())
#print(adk_open.atoms.select_atoms("backbone").timeseries)

#image_size=200
#ref = torch.LongTensor(1,image_size,3).random_(15).float() #coord
#traj = torch.LongTensor(100,image_size,3).random_(15).float() #coords
#trajv = torch.LongTensor(100,image_size,3).random_(15).float() #coords

# monster = Monster()
# monster.forward_intermediates(ref.expand_as(traj), traj)

def train(reference, trajectory, trajectoryv):
    os.chdir("/Scr/hyunpark/Monster/")

    reference, trajectory, trajectoryv = reference, trajectory[:70], trajectoryv[:70]
    min_encoding_size=int(image_size*0.65)
    monster = Monster(n_e=int(min_encoding_size), min_encoding_size=min_encoding_size, image_shape=[image_size]*2, sequence_length=image_size)
    # monster.forward_intermediates(reference.expand_as(trajectory), trajectory)
    dataset = torch.utils.data.TensorDataset(trajectory)   #Only easy trajs (closed states)
    datasetv = torch.utils.data.TensorDataset(trajectoryv) #
    #dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
    #dataloaderv = torch.utils.data.DataLoader(datasetv, batch_size=16, shuffle=True)
    dataset_total = torch.utils.data.ConcatDataset([dataset, datasetv])
    lend = len(dataset)
    lendv = len(datasetv)
    del dataset
    del datasetv
    gc.collect()
    dataset, datasetv = torch.utils.data.random_split(dataset_total, [lend + lendv - 40, 40])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
    dataloaderv = torch.utils.data.DataLoader(datasetv, batch_size=16, shuffle=True)    
    del dataset
    del datasetv
    del dataset_total
    gc.collect()

    epochs = 50
    optim = torch.optim.AdamW(monster.parameters())

    monster.to(device)
    for e in range(epochs):
        monster.train()
        valloss = 0.
        for idx, traj in enumerate(dataloader):
            traj = traj[0].to(device)
            monster.train()
            optim.zero_grad()
            loss = monster.forward_intermediates(reference.expand_as(traj).to(device), traj)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(monster.parameters(), 0.7)
            optim.step()
        for traj in dataloaderv:
            traj = traj[0].to(device)
            monster.eval()
            loss = monster.forward_intermediates(reference.expand_as(traj).to(device), traj)
            valloss += loss.item()
        print(fmt.yellow(f"At epoch {e}, mean validation loss is {valloss/len(dataloaderv)}..."))
        torch.save(monster.state_dict(), "ckpt_monster.pt")
    return monster

monster = train(ref, traj, trajv)


