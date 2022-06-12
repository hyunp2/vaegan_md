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
from MDAnalysis.tests.datafiles import CRD, PSF, DCD, DCD2
from MDAnalysis.analysis import align
from MDAnalysis.coordinates.memory import MemoryReader
from MDAnalysis.analysis.base import AnalysisFromFunction

warnings.simplefilter("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_printoptions(precision=4)

##################################
class Generator(torch.nn.Module):
    def __init__(self, **kwargs):
        super(Generator, self).__init__()
        self.latent = kwargs.get("latent", 2) #2D VAE latents
        self.in_feat = kwargs.get("in_feat", self.latent) #Hidden vec
        self.out_feat = kwargs.get("out_feat", 2) #2D VAE latents
        def block(in_feat, out_feat, normalize=True):
            layers = [torch.nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(torch.nn.BatchNorm1d(out_feat, 0.8))
            layers.append(torch.nn.LeakyReLU(inplace=True))
            return layers
        self.add_module("expansion", torch.nn.Linear(self.in_feat, self.latent))
        self.model = torch.nn.Sequential(
            *block(self.latent, self.latent, normalize=False),
            *block(self.latent, self.latent, normalize=False),
            *block(self.latent, self.latent, normalize=False),
            *block(self.latent, self.latent, normalize=False),
            *block(self.latent, self.latent, normalize=False),
            *block(self.latent, self.latent, normalize=False),
            *block(self.latent, self.latent, normalize=False),
            torch.nn.Linear(self.latent, self.out_feat),
            torch.nn.ELU()
        )
    def forward(self, z):
        z = self.expansion(z)
        lv = self.model(z) #2D VAE latent vec from hidden vec of 128-dim
        return lv

class Discriminator(torch.nn.Module):
    def __init__(self, **kwargs):
        super(Discriminator, self).__init__()
        self.in_feat = kwargs.get("in_feat", 2) #2D VAE latents
        self.latent = kwargs.get("latent", 2) #2D VAE latents
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.in_feat, self.latent),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Linear(self.latent, self.latent),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Linear(self.latent, self.latent),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Linear(self.latent, self.latent),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Linear(self.latent, 1),         
        )
    def forward(self, lv):
        validity = self.model(lv)
        return validity #Value, not 0-1

class GAN(torch.nn.Module):
    def __init__(self, **kwargs):
        super(GAN, self).__init__()
        self.latent = kwargs.get("latent", 2) #2D VAE latents
        self.in_feat_critic = kwargs.get("in_feat_critic", 2) #2D VAE latents
        self.in_feat_gen = kwargs.get("in_feat_gen", self.latent) #2D VAE latents
        self.out_feat_gen = kwargs.get("out_feat_gen", 2) #2D VAE latents
        self.critic = Discriminator(in_feat = self.in_feat_critic, latent=self.latent)
        self.generator = Generator(in_feat = self.in_feat_gen, out_feat = self.out_feat_gen, latent=self.latent)

class cGenerator(Generator):
    def __init__(self, **kwargs):
        super(cGenerator, self).__init__(**kwargs)
        self.cond_dim = kwargs.get("cond_dim", 10) 
        self.add_module("expansion", torch.nn.Linear(self.in_feat + self.cond_dim, self.latent))

    def forward(self, z, conds: torch.LongTensor):
        cond = torch.nn.functional.one_hot(torch.LongTensor(conds.to("cpu")), num_classes=self.cond_dim).float().to(z).squeeze()
        #print(cond.size())
        z_ = torch.cat((z, cond), dim=-1)
        z_ = self.expansion(z_)
        lv = self.model(z_)
        return lv

class cDiscriminator(Discriminator):
    def __init__(self, **kwargs):
        super(cDiscriminator, self).__init__(**kwargs)
        self.num_cls = kwargs.get("num_cls", 1)
        self.add_module("critic", torch.nn.Linear(self.latent, 1))
        self.add_module("classifier", torch.nn.Linear(self.latent, self.num_cls))
        num_mods = self.model._modules.keys().__len__()
        delattr(self.model, str(num_mods-1))
    def forward(self, lv, *args):
        pre = self.model(lv)
        validity = self.critic(pre)
        classes = self.classifier(pre)
        return validity, classes.view(lv.size(0), -1) #Value, not 0-1

class cGAN(GAN):
    def __init__(self, **kwargs):
        super(cGAN, self).__init__(**kwargs)
        self.critic_type = kwargs.get("critic_type", 0)
        self.num_cls = kwargs.get("num_cls", 1) #num_cls and cond_dim are same
        self.critic = cDiscriminator(in_feat = self.in_feat_critic, latent=self.latent, num_cls=self.num_cls) if self.critic_type==0 else cDiscriminator2(in_feat = self.in_feat_critic, latent=self.latent, num_cls=self.num_cls)
        self.generator = cGenerator(in_feat = self.in_feat_gen, out_feat = self.out_feat_gen, latent=self.latent, cond_dim=self.num_cls)


