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
import run_monster_vae as vae
import time
import sklearn.cluster

warnings.simplefilter("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_printoptions(precision=4)

###########################
##########Data#############
###########################
from MDAnalysis.tests.datafiles import CRD, PSF, DCD, DCD2
from MDAnalysis.analysis import align
from MDAnalysis.coordinates.memory import MemoryReader
from MDAnalysis.analysis.base import AnalysisFromFunction
warnings.simplefilter("ignore")
#torch.manual_seed(42)

adk = mda.Universe("reference.pdb")
#adk_open = mda.Universe(CRD, DCD2)
#adk_closed = mda.Universe(PSF, DCD)
ADKs = mda.Universe("reference.pdb", "adk.dcd")

"""
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
"""

atom_selection = "backbone"
#copen = AnalysisFromFunction(lambda ag: ag.positions.copy(),
#                                   adk_open.atoms.select_atoms(f"{atom_selection}")).run().results['timeseries']
#cclosed = AnalysisFromFunction(lambda ag: ag.positions.copy(),
#                                   adk_closed.atoms.select_atoms(f"{atom_selection}")).run().results['timeseries']
reference = torch.from_numpy(adk.atoms.select_atoms(f"{atom_selection}").positions)[None,:]
coords = AnalysisFromFunction(lambda ag: ag.positions.copy(),
                                   ADKs.atoms.select_atoms(f"{atom_selection}")).run().results['timeseries']
trajectory = torch.from_numpy(coords[:102]) #Test
trajectoryv = torch.from_numpy(coords[102:]) #Validation
#image_size = trajectory.size(1) #Sequence length

reference = reference.to(device)
trajectory = trajectory.to(device) 
trajectoryv = trajectoryv.to(device)

def normalize(coords, mean=None, std=None):
    coords = coords.view(coords.size(0), -1)
    mean = coords.mean(dim=0) if mean == None else mean #(B,C)
    std = coords.std(dim=0) if std == None else std #(B,C)
    coords_ = (coords - mean) / std
    coords_ = coords_.view(coords.size(0), coords.size(1)//3 ,3) #Back to original shape (B,L,3)
    return coords_, mean, std

def unnormalize(coords, mean=None, std=None):
    assert mean != None and std != None, "Wrong arguments..."
    coords = coords.view(coords.size(0), -1)
    coords_ = (coords * std) + mean
    coords_ = coords_.view(coords.size(0), coords.size(1)//3 ,3) #Back to original shape (B,L,3)
    return coords_

trajectory, mean, std = normalize(trajectory, mean=None, std=None)
trajectoryv, _, _ = normalize(trajectoryv, mean=mean, std=std)

###############################
##############VAE##############
###############################
model = vae.VAE(hidden_dims_enc=[1500, 1000, 750, 300, 4], hidden_dims_dec=[2, 300, 750, 1000, 1500], reference=reference, rolled_dim=3, unrolled_dim=trajectory.size(1)*trajectory.size(2), mha_dimension=1200, nheads=6, layers=8)
ckpt = torch.load("New_VAE.pt", map_location=device)
model.load_state_dict(ckpt)
model.to(device)
model.eval()

def lerp(inputs: "1D tensor of features", outputs: "1D tensor of features", interps: "1D tensor of weights"):
    outs = inputs + (outputs - inputs) * interps.view(-1,1).to(inputs)
    return outs

traj = torch.cat((trajectory, trajectoryv), dim=0)
z, mu, logstd, recon = model(traj)

dataloader = torch.utils.data.DataLoader(traj, batch_size=32, shuffle=True) #Normalized trajectory! BL3
###############################
##############GAN##############
###############################
#https://jonathan-hui.medium.com/gan-wasserstein-gan-wgan-gp-6a1a2aa1b490
#https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan_gp/wgan_gp.py

def compute_gradient_penalty(D, real_samples, fake_samples, device):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand(real_samples.size(0), 1).to(device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True).to(device)
    d_interpolates = D(interpolates)
    fake = torch.ones(real_samples.shape[0], 1).float().to(device)
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=-1) - 1) ** 2).mean()
    return gradient_penalty

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


###############################
###########Training############
###############################
def training(epochs = 10, n_critic=3, sample_size = 128, dataloader = dataloader, lambda_gp=2, every=10, adams=None, latent=6, save_to=None, struct: "BL3"=None):
    # Initialize generator and discriminator
    unrolled_size = struct.size(-2)*struct.size(-1)
    gan = GAN(latent=latent, in_feat_critic=unrolled_size, in_feat_gen=sample_size, out_feat_gen=unrolled_size)
    generator = gan.generator.to(device)
    discriminator = gan.critic.to(device)
    lr = adams.get("lr", None)
    betas = adams.get("betas", None) 
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=betas)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=betas)
    
    for epoch in range(epochs):
        generator.train()
        discriminator.train()
        for c in range(n_critic):
            generator.train()
            discriminator.train()
            optimizer_D.zero_grad()
            for i, imgs in enumerate(dataloader):            
                imgs = imgs.to(device) 
                # Configure input
                #print(imgs.size())
                real_imgs = imgs.view(-1, unrolled_size) #Unroll to B by 3L
                # ---------------------
                #  Train Discriminator
                # ---------------------                
                # Sample noise as generator input
                z = torch.randn(imgs.shape[0], sample_size).to(imgs)
                # Generate a batch of images
                [param.requires_grad_(False) for param in generator.parameters()]
                fake_imgs = generator(z)
                #print(imgs.size(), real_imgs.size(), fake_imgs.size())
                # Real images
                real_validity = discriminator(real_imgs)
                # Fake images
                fake_validity = discriminator(fake_imgs)
                # Gradient penalty
                gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.clone().detach(), fake_imgs.clone().detach(), device)
                # Adversarial loss
                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty #Wasserstein distance: big negative means discriminator is too good? Should it be near 0 upon convergence???
                d_loss.backward(retain_graph=True)
                [param.requires_grad_(True) for param in generator.parameters()]
            optimizer_D.step()
        optimizer_G.zero_grad()
        # Train the generator every n_critic steps
        #if i % n_critic == 0:
        # -----------------
        #  Train Generator
         # -----------------
        # Generate a batch of images
        z = torch.randn(imgs.shape[0], sample_size).to(imgs)
        fake_imgs = generator(z)
        # Loss measures generator's ability to fool the discriminator
        # Train on fake images
        [param.requires_grad_(False) for param in discriminator.parameters()]
        fake_validity = discriminator(fake_imgs)
        g_loss = -torch.mean(fake_validity) #Big negative means discriminator is fooled by good generator samples
        g_loss.backward()
        optimizer_G.step()
        [param.requires_grad_(True) for param in discriminator.parameters()]
        print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, epochs, i, len(dataloader), d_loss.item(), g_loss.item())
                )
        if epoch % every == 0:
            generator.eval()
            z = torch.randn(1000, sample_size).to(imgs)
            with torch.no_grad():
                fake_imgs = generator(z)
                _, fake_mu, *_ = model.encoder(fake_imgs)
            plt.scatter(*fake_mu.detach().cpu().numpy().T)
            plt.scatter(*mu.detach().cpu().numpy().T)  
            plt.title(f"gan_{epoch}")          
            plt.savefig(f"gan_pics/gan_{epoch}.png"); 
            plt.close()
        torch.save(gan.state_dict(), f"{save_to}")

training_parmas = dict(epochs = 10001, n_critic=20, dataloader = dataloader, lambda_gp=10, every=250, adams=dict(lr=0.0002, betas=(0.5,0.9)), latent=1000, sample_size = 512, save_to="GAN.pt", struct=next(iter(dataloader))) #Need a VERY good discriminator first and foremost!!
#training(**training_parmas)


###############################
########Evaluation: GAN########
###############################
def evaluate():
    unrolled_size = next(iter(dataloader)).size(1) * next(iter(dataloader)).size(2)
    sample_size = 512
    gan = GAN(latent=1000, in_feat_critic=unrolled_size, in_feat_gen=sample_size, out_feat_gen=unrolled_size).to(device)
    ckpt = torch.load("GAN.pt", map_location=device)
    gan.load_state_dict(ckpt)
    gen = gan.generator
    gen.to(device)
    gen.eval()

    imgs = next(iter(dataloader)).to(device)
    z = torch.randn(500, sample_size).to(imgs)
    fake_imgs = gen(z)
    _, fake_mu, *_ = model.encoder(fake_imgs)
    #plt.scatter(*fake_mu.detach().cpu().numpy().T)
    #plt.scatter(*mu.detach().cpu().numpy().T)  
    #plt.show()
    reconed = model.decoder(fake_mu)
    unnorm = unnormalize(reconed, mean=mean, std=std)

    gan_closed = mda.Universe(CRD)
    gan_closed_ca = gan_closed.atoms.select_atoms(atom_selection)
    gan_closed_ca.atoms.positions = unnorm[30].detach().cpu().numpy()
    gan_closed_ca.write("gan0.pdb")
#evaluate()

###############################
##############cGAN#############
###############################
def c_compute_gradient_penalty(D, real_samples, fake_samples, device, *args, **kwargs):
    critic_type = kwargs.get("critic_type", 0)
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand(real_samples.size(0), 1).to(device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True).to(device)
    if critic_type == 0:
        d_interpolates, *_ = D(interpolates, *args)
    else:
        d_interpolates = D(interpolates, *args)
    fake = torch.ones(real_samples.shape[0], 1).float().to(device)
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=-1) - 1) ** 2).mean()
    return gradient_penalty

class cGenerator(Generator):
    def __init__(self, **kwargs):
        super(cGenerator, self).__init__(**kwargs)
        self.cond_dim = kwargs.get("cond_dim", 10) 
        self.add_module("expansion", torch.nn.Linear(self.in_feat + self.cond_dim, self.latent))
        #def block(in_feat, out_feat, normalize=True):
        #    layers = [torch.nn.Linear(in_feat, out_feat)]
        #    if normalize:
        #        layers.append(torch.nn.BatchNorm1d(out_feat, 0.8))
        #    layers.append(torch.nn.LeakyReLU(inplace=True))
        #    return torch.nn.Sequential(*layers)
        #self.lv = torch.nn.ModuleList([
        #    block(self.in_feat, self.latent, normalize=False),
        #    block(self.latent, self.latent, normalize=False),
        #    block(self.latent, self.latent, normalize=False),
        #    block(self.latent, self.latent, normalize=False),
        #    block(self.latent, self.latent, normalize=False),
        #    block(self.latent, self.latent, normalize=False),]
        #)
        #self.add_module("condition", torch.nn.Sequential(
        #    *block(self.cond_dim, self.latent, normalize=False),))
        #self.add_module("combine", torch.nn.Sequential(torch.nn.Linear(self.latent, self.out_feat),
        #                                               torch.nn.Tanh()))
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

class cDiscriminator2(Discriminator):
    def __init__(self, **kwargs):
        super(cDiscriminator2, self).__init__(**kwargs)
        self.num_cls = kwargs.get("num_cls", 1)
        self.add_module("critic", torch.nn.Linear(self.latent, 1))
        self.model[0] = torch.nn.Linear(self.in_feat + self.num_cls, self.latent)
        num_mods = self.model._modules.keys().__len__()
        delattr(self.model, str(num_mods-1))
    def forward(self, lv, *args):
        cond = args[0]
        cond = torch.nn.functional.one_hot(torch.LongTensor(cond.to("cpu")), num_classes=self.num_cls).float().to(z).squeeze()
        lv = torch.cat((lv, cond), dim=-1)
        pre = self.model(lv)
        validity = self.critic(pre)
        return validity

class cGAN(GAN):
    def __init__(self, **kwargs):
        super(cGAN, self).__init__(**kwargs)
        self.critic_type = kwargs.get("critic_type", 0)
        self.num_cls = kwargs.get("num_cls", 1) #num_cls and cond_dim are same
        self.critic = cDiscriminator(in_feat = self.in_feat_critic, latent=self.latent, num_cls=self.num_cls) if self.critic_type==0 else cDiscriminator2(in_feat = self.in_feat_critic, latent=self.latent, num_cls=self.num_cls)
        self.generator = cGenerator(in_feat = self.in_feat_gen, out_feat = self.out_feat_gen, latent=self.latent, cond_dim=self.num_cls)

km = sklearn.cluster.KMeans(n_clusters=5, random_state=42) #Pseudo MSM
km.fit(mu.detach().cpu().numpy())
#plt.scatter(*mu.T.detach().cpu().numpy(), c=km.labels_, s=10); plt.show()
labels = np.unique(km.labels_)

cdataset = torch.utils.data.TensorDataset(traj, torch.from_numpy(km.labels_).long().view(-1,1)) #traj: BL3; km.labels_: B1
cdataloader = torch.utils.data.DataLoader(cdataset, batch_size=32, shuffle=True)
#print(next(iter(cdataloader)))

###############################
###########Training############
###############################
def ctraining(epochs = 10, n_critic=3, dataloader = dataloader, sample_size = 128, lambda_gp=2, lambda_ce=1, every=10, adams=None, latent=6, save_to=None, num_cls=6, struct: "BL3"=None, critic_type=0):
    device = torch.device("cuda:15")
    # Initialize generator and discriminator
    unrolled_size = struct.size(-2)*struct.size(-1)

    gan = cGAN(latent=latent, in_feat_critic=unrolled_size, in_feat_gen=sample_size, out_feat_gen=unrolled_size, num_cls=num_cls, critic_type=critic_type)
    generator = gan.generator.to(device)
    discriminator = gan.critic.to(device)
    lr = adams.get("lr", None)
    betas = adams.get("betas", None) 
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=betas)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=betas)

    for epoch in range(epochs):
        generator.train()
        discriminator.train()
        for c in range(n_critic):
            generator.train()
            discriminator.train()
            optimizer_D.zero_grad()
            for i, imgs_ in enumerate(dataloader):            
                imgs = imgs_[0].to(device)
                conds = imgs_[1].to(device)

                # Configure input
                real_imgs = imgs.view(-1, unrolled_size) #Unroll to B by 3L
                #print(real_imgs.size())
                # ---------------------
                #  Train Discriminator
                # ---------------------                
                # Sample noise as generator input
                z = torch.randn(imgs.shape[0], sample_size).to(imgs)
                # Generate a batch of images
                [param.requires_grad_(False) for param in generator.parameters()]
                fake_imgs = generator(z, conds)
                #print(imgs.size(), real_imgs.size(), fake_imgs.size())
                # Real images
                if critic_type == 0:
                    real_validity, real_cls = discriminator(real_imgs)
                    # Fake images
                    fake_validity, *_ = discriminator(fake_imgs)
                else:
                    real_validity = discriminator(real_imgs, conds)
                    #print(real_validity.size(), conds.size())
                    # Fake images
                    fake_validity = discriminator(fake_imgs, conds)

                # Gradient penalty
                gradient_penalty = c_compute_gradient_penalty(discriminator, real_imgs.clone().detach(), fake_imgs.clone().detach(), device, conds, critic_type=critic_type)
                # Adversarial loss
                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty #Wasserstein distance: big negative means discriminator is too good? Should it be near 0 upon convergence???
                d_loss = d_loss + lambda_ce * torch.nn.CrossEntropyLoss(reduction="mean")(real_cls, conds.view(-1,)) if critic_type==0 else d_loss
                d_loss.backward(retain_graph=True)
                [param.requires_grad_(True) for param in generator.parameters()]
            optimizer_D.step()
        optimizer_G.zero_grad()
        # Train the generator every n_critic steps
        #if i % n_critic == 0:
        # -----------------
        #  Train Generator
         # -----------------
        # Generate a batch of images
        z = torch.randn(imgs.shape[0], sample_size).to(imgs)
        fake_conds = torch.LongTensor(conds.size(0)).random_(num_cls).to(device)

        fake_imgs = generator(z, fake_conds)
        # Loss measures generator's ability to fool the discriminator
        # Train on fake images
        [param.requires_grad_(False) for param in discriminator.parameters()]
        if critic_type == 0:
            fake_validity, fake_cls = discriminator(fake_imgs)
        else:
            fake_validity = discriminator(fake_imgs, fake_conds)
        g_loss = -torch.mean(fake_validity) #Big negative means discriminator is fooled by good generator samples
        #print(fake_conds.view(-1,))
        g_loss = g_loss + lambda_ce * torch.nn.CrossEntropyLoss(reduction="mean")(fake_cls, fake_conds.view(-1,)) if critic_type==0 else g_loss
        g_loss.backward()
        optimizer_G.step()
        [param.requires_grad_(True) for param in discriminator.parameters()]
        print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, epochs, i, len(dataloader), d_loss.item(), g_loss.item())
                )
        if epoch % every == 0:
            generator.eval()
            with torch.no_grad():
                z = torch.randn(1000, sample_size).to(imgs)
                fake_conds = torch.LongTensor(1000).random_(num_cls).to(device)
                fake_imgs = generator(z, fake_conds)
                _, fake_mu, *_ = model.encoder(fake_imgs)
            plt.scatter(*fake_mu.detach().cpu().numpy().T, c=fake_conds.detach().cpu().numpy().reshape(-1,))
            plt.scatter(*mu.detach().cpu().numpy().T, c='w', edgecolors=km.labels_, s=25)  
            if critic_type == 0:
                plt.title(f"cgan_{epoch}")          
                plt.savefig(f"cgan_pics/cgan_{epoch}.png"); 
                plt.close()
                torch.save(gan.state_dict(), f"{save_to}")
            else:
                plt.title(f"cgan_{epoch}")          
                plt.savefig(f"cgan_pics_2/cgan_{epoch}.png"); 
                plt.close()
                torch.save(gan.state_dict(), f"2_{save_to}")

#ctraining_parmas = dict(epochs = 10001, n_critic=20, dataloader = cdataloader, lambda_gp=10, lambda_ce=2, every=250, adams=dict(lr=0.0002, betas=(0.5,0.9)), latent=128, save_to="cGAN.pt", num_cls=labels[-1]+1) #Need a VERY good discriminator first and foremost!!
ctraining_parmas = dict(epochs = 10001, n_critic=20, dataloader = cdataloader, lambda_gp=10, lambda_ce=2, every=250, adams=dict(lr=0.0002, betas=(0.5,0.9)), latent=1000, sample_size = 512, save_to="cGAN.pt", num_cls=labels[-1]+1, struct=next(iter(cdataloader))[0], critic_type=1)
#ctraining(**ctraining_parmas)

###############################
########Evaluation: cGAN#######
###############################
def lerp(inputs: "1D tensor of features", outputs: "1D tensor of features", interps: "1D tensor of weights"):
    outs = inputs + (outputs - inputs) * interps.view(-1,1).to(inputs)
    return outs

def cevaluate():
    unrolled_size = next(iter(dataloader)).size(1) * next(iter(dataloader)).size(2)
    interpolations = 100
    trials = 10
    num_samples = 10000
    sample_size = 512
    latent = 1000
    num_cls = 5
    critic_type=1
    device = torch.device("cuda:5")
    gan = cGAN(latent=latent, in_feat_critic=unrolled_size, in_feat_gen=sample_size, out_feat_gen=unrolled_size, num_cls=num_cls, critic_type=critic_type).to(device)
    ckpt = torch.load("2_cGAN.pt", map_location=device)
    gan.load_state_dict(ckpt, strict=False)
    gen = gan.generator
    gen.to(device)
    gen.eval()
    model.to(device)

    with torch.no_grad():
        z = torch.randn(num_samples, sample_size).to(device)
        #fake_conds = torch.LongTensor(num_samples).random_(num_cls).to(device)
        #fake_conds = torch.repeat_interleave(torch.LongTensor([0,1,2,3,4]), num_samples//num_cls)
        fake_conds = torch.repeat_interleave(torch.LongTensor([0,3]), num_samples//2)
        fake_imgs = gen(z, fake_conds)
    _, fake_mu, *_ = model.encoder(fake_imgs)
    #print(plt.cm.get_cmap("jet")(km.labels_))
    plt.scatter(*mu.detach().cpu().numpy().T, c=km.labels_.reshape(-1,), s=50)  
    [plt.annotate(str(labels[i]), km.cluster_centers_[i], bbox=dict(boxstyle="round", fc='w')) for i in range(len(labels))]


    for t in range(trials):
        z = torch.randn(1, sample_size).to(device).expand(interpolations,-1)
        fake_conds = torch.LongTensor([0,1,2,3,4])
        cond = torch.nn.functional.one_hot(torch.LongTensor(fake_conds.to("cpu")), num_classes=num_cls).float().to(z).squeeze()
        lerps = lerp(cond[2], cond[1], torch.linspace(0,1,interpolations))
        z_ = torch.cat((z, lerps), dim=-1)
        z_ = gen.expansion(z_)
        lv = gen.model(z_)
        _, fake_mu, *_ = model.encoder(lv)
        plt.scatter(*fake_mu.detach().cpu().numpy().T, c='r', s=25, alpha=0.8)

    #plt.scatter(*fake_mu.detach().cpu().numpy().T, c=fake_conds.detach().cpu().numpy().reshape(-1,), s=15, alpha=0.8)
    plt.colorbar()
    plt.savefig("cgan0_nothing.png")
    reconed = model.decoder(fake_mu)
    unnorm = unnormalize(reconed, mean=mean.to(device), std=std.to(device))

    gan_closed = mda.Universe(CRD)
    gan_closed_ca = gan_closed.atoms.select_atoms(atom_selection)
    gan_closed_ca.atoms.positions = unnorm[0].detach().cpu().numpy()
    gan_closed_ca.write("cgan0.pdb")
cevaluate()

