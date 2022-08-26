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
import copy
import pytorch_lightning as pl
from MDAnalysis.analysis import align
from MDAnalysis.coordinates.memory import MemoryReader
from MDAnalysis.analysis.base import AnalysisFromFunction
import argparse

warnings.simplefilter("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_printoptions(precision=4)

##################################
class Encoder(torch.nn.Module):
    def __init__(self, hidden_dims=[1000, 500, 100, 50, 4], **kwargs):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.unrolled_dim = kwargs.get("unrolled_dim") #xyz coord dim of original protein trajectory

        linears = torch.nn.Sequential(*[ 
                                      torch.nn.Linear(self.unrolled_dim, self.hidden_dims[0]), torch.nn.SiLU(), 
                                      torch.nn.Linear(self.hidden_dims[0], self.hidden_dims[1]), torch.nn.SiLU(True),                 
                                      torch.nn.Linear(self.hidden_dims[1], self.hidden_dims[2]), torch.nn.SiLU(True),
                                      torch.nn.Linear(self.hidden_dims[2], self.hidden_dims[3]), torch.nn.SiLU(True),                
                                      torch.nn.Linear(self.hidden_dims[3], self.hidden_dims[4]),                     
                                    ]) #B,2
        self.add_module("linears_sequential", linears)

    def forward(self, inputs):
        sizes = inputs.size()
        x = inputs #BLC -> Cartesian coords...
        x = x.view(sizes[0], -1)
        x = self.linears_sequential(x)
        mu, logstd = torch.chunk(x, 2, dim=-1)
        z = self.reparameterize(mu, logstd)
        return z, mu, logstd

    def reparameterize(self, mu, logstd):
        shapes = mu.shape
        return mu + logstd.exp() * torch.distributions.Normal(0., 0.1).rsample((shapes)).to(mu)

class Decoder(torch.nn.Module):
    def __init__(self, hidden_dims=list(reversed([2, 50, 100, 500, 1000])), **kwargs):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.unrolled_dim = kwargs.get("unrolled_dim") #xyz coord dim of original protein trajectory
#         self.rolled_dim = kwargs.get("rolled_dim", None) #xyz coord dim of original protein trajectory
#         self.reference = kwargs.get("reference", None) #PDB of reference
#         self.mha_dimension = kwargs.get("mha_dimension", None)
#         self.nheads = kwargs.get("nheads", None)
#         self.layers = kwargs.get("layers", None)

        linears = torch.nn.Sequential(*[ 
                                      torch.nn.Linear(self.hidden_dims[0], self.hidden_dims[1]), torch.nn.SiLU(True),                 
                                      torch.nn.Linear(self.hidden_dims[1], self.hidden_dims[2]), torch.nn.SiLU(True),
                                      torch.nn.Linear(self.hidden_dims[2], self.hidden_dims[3]), torch.nn.SiLU(True),                
                                      torch.nn.Linear(self.hidden_dims[3], self.hidden_dims[4]), torch.nn.SiLU(True),           
                                      torch.nn.Linear(self.hidden_dims[4], self.unrolled_dim)
                                    ]) #B,C,H,W
        self.add_module("linears_sequential", linears)
        #self.mha_res = MultiheadAttention_Residual(rolled_dim=self.rolled_dim, mha_dimension=self.mha_dimension, nheads=self.nheads)
        #feedforward = torch.nn.Sequential(*[torch.nn.Linear(self.rolled_dim, self.rolled_dim), torch.nn.LeakyReLU(True), torch.nn.Linear(self.rolled_dim, self.rolled_dim)])
        #self.add_module("ff", feedforward)
        #self.pos_emb = torch.nn.Embedding(self.reference.size(1), 3) #reference is (1,L,3)

    def forward(self, inputs: "BD"):
        sizes = (1, self.unrolled_dim//3, 3) #1,L,3
        x = inputs #Latent dim
        x = self.linears_sequential(x)
#         x = 3*torch.tanh(x)
        x_q = x.view(x.size(0), sizes[1], sizes[2]) #+ pos_emb #B,L,3 
        return x_q #, attns


class VAE(torch.nn.Module):
    #VAE github: https://github.com/AntixK/PyTorch-VAE/tree/master/models
    """Input and output are both (B,L,3) and flattend inside Encoder/Decoder!"""
    def __init__(self, args: argparse.ArgumentParser, **kwargs):
        super().__init__()
        self.args = args
        self.hidden_dims_enc = kwargs.get("hidden_dims_enc", None)
        self.hidden_dims_dec = kwargs.get("hidden_dims_dec", None)
        self.unrolled_dim = kwargs.get("unrolled_dim", None) #xyz coord dim of original protein trajectory
#         self.rolled_dim = kwargs.get("rolled_dim") #xyz coord dim of original protein trajectory
#         self.reference = kwargs.get("reference") #PDB of reference
#         self.mha_dimension = kwargs.get("mha_dimension", 1200)
#         self.nheads = kwargs.get("nheads", 6)
#         self.layers = kwargs.get("layers", 6)
        self.encoder = Encoder(hidden_dims=self.hidden_dims_enc, unrolled_dim=self.unrolled_dim)
#         self.decoder = Decoder(hidden_dims=self.hidden_dims_dec, reference=self.reference, rolled_dim=self.rolled_dim, unrolled_dim=self.unrolled_dim, mha_dimension=self.mha_dimension, nheads=self.nheads, layers=self.layers)
        self.decoder = Decoder(hidden_dims=self.hidden_dims_dec, unrolled_dim=self.unrolled_dim)
#         self.apply(self._init_weights)
        self.reset_all_weights()
    
    def forward(self, inputs: "Trajectory"):
        x = inputs #Normalized input
        z, mu, logstd = self.encoder(x)
        x = self.decoder(z) #BL3, Dict: BHLL x Layers
        return z, mu, logstd, x
    
    @staticmethod
    def losses(inputs, z, mu, logstd, recon: "x", beta):
#         rmsd = torch.sqrt(torch.mean((inputs - recon)**2, dim=(-1, -2))).mean() #rmsd
        mse = torch.nn.MSELoss(reduction="none")(recon, inputs).sum(dim=(1,2)) # -> (B,)
        kl = beta * 0.5 * torch.sum(1 + logstd - mu ** 2 - logstd.exp(), dim = 1)  #kl-div (NOT a LOSS yet!); -> (B,)
#         L = max(15, inputs.shape[-2])
#         d0 = 1.24 * (L - 15)**(1/3) - 1.8
#         # get distance
#         dist = ((inputs - recon)**2).sum(dim=-1).sqrt()
#         tm = (1 / (1 + (dist/d0)**2)).mean(dim=-1).mean() #TM-score
#         inputs_mat = torch.cdist(inputs, inputs, p=2)
#         recon_mat = torch.cdist(recon, recon, p=2)
#         mat = 0.5*(inputs_mat - recon_mat).pow(2).sum(dim=(-1,-2)).mean() #Pairwise distance loss
#         return kl, mse, rmsd, tm, mat
#         print(mse.size(), kl.size())
        assert mse.size(0) == kl.size(0) and mse.ndim == kl.ndim and mse.ndim == 1, "all criteria for shape must match"
        return mse, kl

    def reset_all_weights(self, ) -> None:
        """
        refs:
        - https://discuss.pytorch.org/t/how-to-re-set-alll-parameters-in-a-network/20819/6
        - https://stackoverflow.com/questions/63627997/reset-parameters-of-a-neural-network-in-pytorch
        - https://pytorch.org/docs/stable/generated/torch.nn.Module.html
        """

        @torch.no_grad()
        def weight_reset(m: torch.nn.Module):
             # - check if the current module has reset_parameters & if it's callabed called it on m
            reset_parameters = getattr(m, "reset_parameters", None)
            if callable(reset_parameters):
                m.reset_parameters()

        # Applies fn recursively to every submodule see: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
        self.apply(fn=weight_reset)

