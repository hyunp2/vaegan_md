import torch, torchani
import transformers as tfm
import dataloader as dl 
import physnet_block as pb
import pytorch_lightning as pl
import losses as ls
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import MDAnalysis as mda
import tqdm
import pickle, json, os, copy
import gc
import collections
import networkx as nx
import yaml
import logging
import warnings
import wandb


np.random.seed(42)
torch.manual_seed(42)

logging.basicConfig()
logger = logging.getLogger("model.py logger")
logger.setLevel(logging.DEBUG)

warnings.simplefilter("ignore")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Model(pl.LightningModule):
    def __init__(self, args, **kwargs):
        super().__init__()
        self.which_emodel = kwargs.get("which_emodel", "physnet") #useless
        assert self.which_emodel == "physnet", "Legacy code supports only Physnet..." #useless 
        energy_config = kwargs.get("physnet_block")
        self.energy_block = pb.Physnet(**energy_config)     
        self.bonds = self.energy_block.bonds
        self.coord_ref = self.energy_block.coord_ref
        self.species = self.energy_block.species
        self.spectral_norm = kwargs.get("spectral_norm", False)
        if self.spectral_norm:
            self.spectral_normalize(self.energy_block)
        self.mbuffer = ls.MemoryBuffer()
        self.losses = dict(train_boltzmann=[], val_boltzmann=[], train_contrastive=[], val_contrastive=[])
        self.loss_type = args.loss_type
        self.lr = args.lr
        self.scheduler = args.scheduler
        self.loader_length = args.loader_length
        self.args = args
        wandb.init(project="EBM", entity="hyunp2")
        self.wandb_table = wandb.Table(columns=["batch_idx", "pairs0", "pairs4"])
        self.counter_pred = 0
        wandb.watch(self.energy_block, log="all")
        self.ignore_pair = False

    def freeze_gradient(self, module):
        for m in module.modules():
            m.requires_grad = False

    def spectral_normalize(self, module: torch.nn.Module):
        for m in module.modules():
            try:
                torch.nn.utils.parametrizations.spectral_norm(m)    
            except Exception as e:
                print(f"{e} at {m}... skipping...")      
      
    def forward(self, inputs):
        coords = inputs #species and coords
        x, pair = self.energy_block(coords)
        if self.ignore_pair:
            return x
        else:
            return x, pair

    def training_step(self, batch, batch_idx):
        coords = batch
        negs = self.fetch_negatives(coords, self.species, self.energy_block)
        loss_ = self.custom_loss(stage="train")(coords, negs, self.species)
        loss = loss_["boltzmann"] + loss_["contrastive"]
        self.losses["train_boltzmann"].append(loss_["boltzmann"].detach())
        self.losses["train_contrastive"].append(loss_["contrastive"].detach())
        #self.log("train_loss", loss.detach(), prog_bar=True, on_step=True)
        #self.log("train_lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True, on_step=True)
        wandb.log({'train_loss_boltzmann': loss_["boltzmann"].item()})
        wandb.log({'train_loss_contrastive': loss_["contrastive"].item()})
        wandb.log({'train_lr': self.trainer.optimizers[0].param_groups[0]["lr"]})
        return loss

    def validation_step(self, batch, batch_idx):
        loss_ = self._shared_eval_step(batch, batch_idx)
        loss = loss_["boltzmann"] + loss_["contrastive"]
        self.losses["val_boltzmann"].append(loss_["boltzmann"].detach())
        self.losses["val_contrastive"].append(loss_["contrastive"].detach())
        #self.log("val_loss", loss.detach(), prog_bar=True, on_step=True)
        wandb.log({'val_loss_boltzmann': loss_["boltzmann"].item()})
        wandb.log({'val_loss_contrastive': loss_["contrastive"].item()})

    def validation_epoch_end(self, validation_step_outputs):
        if not self.trainer.running_sanity_check:
            #result_dict = {
            #    "epoch": self.current_epoch,
            #    "epoch_lr": self.trainer.optimizers[0].param_groups[0]["lr"],
            #    "epoch_train_loss": torch.stack(self.losses["train_boltzmann"]).mean(),
            #    "epoch_val_loss": torch.stack(self.losses["val_boltzmann"]).mean(),
            #}

            #self.log("epoch", self.current_epoch, prog_bar=True)
            #self.log("epoch_lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True)
            self.log("epoch_train_loss_contrastive", torch.stack(self.losses["train_contrastive"]).mean(), prog_bar=True)
            self.log("epoch_val_loss_contrastive", torch.stack(self.losses["val_contrastive"]).mean(), prog_bar=True)
            wandb.log({'epoch': self.current_epoch})
            wandb.log({'epoch_lr': self.trainer.optimizers[0].param_groups[0]["lr"]})
            wandb.log({'epoch_train_loss_boltzmann': torch.stack(self.losses["train_boltzmann"]).mean()})
            wandb.log({'epoch_val_loss_boltzmann': torch.stack(self.losses["val_boltzmann"]).mean()})
            wandb.log({'epoch_train_loss_contrastive': torch.stack(self.losses["train_contrastive"]).mean()})
            wandb.log({'epoch_val_loss_contrastive': torch.stack(self.losses["val_contrastive"]).mean()})
        self.losses = dict(train_boltzmann=[], val_boltzmann=[], train_contrastive=[], val_contrastive=[])

    def test_step(self, batch, batch_idx):
        loss_ = self._shared_eval_step(batch, batch_idx)
        loss = loss_["boltzmann"] + loss_["contrastive"]
        metrics = {"test_loss": loss.detach()}
        #self.log("test_loss", loss.detach(), prog_bar=True, on_step=True)
        wandb.log({'test_loss_boltzmann': loss_["boltzmann"].item()})
        wandb.log({'test_loss_contrastive': loss_["contrastive"].item()})

    def _shared_eval_step(self, batch, batch_idx):
        coords = batch
        negs = self.fetch_negatives(coords, self.species, self.energy_block)
        loss = self.custom_loss(stage=None)(coords, negs, self.species) 
        return loss #dictionary

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        self.counter_pred += 1
        coords = batch
        negs = self.fetch_negatives(coords, self.species, self.energy_block)
        energyP, pairs = self(coords)
        energyN, *_ = self(negs)
        partition_function_for_P = -torch.cat((energyP, energyN.view(1,-1).expand(energyP.size(0), -1)), dim=-1) #B, 1+B
        log_probP = (-energyP) - torch.logsumexp(partition_function_for_P, dim=1, keepdim=True)
        probs = (log_probP).exp()
        #probs = ls.Losses(model=self.energy_block, loss_type="boltzmann", sigma=1.0, alpha=1.0, kl_loss=False)(coords, negs, self.species)
        wandb.log({"energy": energyP})
        wandb.log({"probs": probs})
        p0 = pairs["G0"][0,...,-1]
        p4 = pairs["G4"][0,...,-1]
        self.wandb_table.add_data(self.counter_pred, wandb.Image(p0), wandb.Image(p4))

    def on_predict_end(self, ):
        wandb.run.log({"Table": self.wandb_table})

    def configure_optimizers(self):
        optimizer = tfm.AdamW(self.energy_block.parameters(), lr=self.lr)
        total_training_steps = self.loader_length * self.args.num_epochs
        warmup_steps = total_training_steps // self.args.warm_up_split
        if self.scheduler == "cosine":
            scheduler = tfm.get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps = warmup_steps, num_training_steps=total_training_steps, num_cycles = 1, last_epoch = -1)
        elif self.scheduler == "linear":
            scheduler = tfm.get_linear_schedule_with_warmup(optimizer, num_warmup_steps = warmup_steps, num_training_steps=total_training_steps, last_epoch = -1)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def custom_loss(self, stage="train"):
        if stage == "train":
            return ls.Losses(model=self.energy_block, loss_type=self.loss_type, sigma=1.0, alpha=1.0, kl_loss=False)
        else:
            return ls.Losses(model=self.energy_block, loss_type=self.loss_type, sigma=1.0, alpha=1.0, kl_loss=False)

    def fetch_negatives(self, coords, species, model):
        if len(self.mbuffer) > coords.size(0):
            fetched_negs = self.mbuffer.sample(coords.size(0)).to(coords) #BLC, Non-differentiable, CPU
            lang = ls.LangevinDynamics(dt=1e-6, temperature=300, intermediates=False)
            neg_imgs = lang(fetched_negs, species, model) #List of non-differentiable intermediate CPUs, if argument passed; last image is differentiable GPU
            negs = neg_imgs.to(coords)  #Chosen image is differentiable GPU
            self.mbuffer +  negs #Save to memory buffer
        else:
            lang = ls.LangevinDynamics(dt=1e-6, temperature=300, intermediates=False)
            neg_imgs = lang(coords, species, model)  #List of non-differentiable intermediate CPUs, if argument passed; last image is differentiable GPU
            negs = neg_imgs.to(coords) #Chosen image is differentiable GPU
            self.mbuffer +  negs #Save to memory buffer   
        return negs                 


if __name__ == '__main__':
    kwargs.update(which_emodel="physnet", ani_block=dict(), schnet_block=dict(), physnet_block=dict(), spectral_norm=False)
    #total_dataset = dl.multiprocessing_dataset()
    #dataset = dl.TorchDataset(total_dataset, 'z15')
    model = Model(**kwargs)

    #dataloaded = dl.DataLoader(dataset, 128).dataloader
    dataloaded = dl.DataLoader(batch_size=100, protein_name="pentapeptide").dataloader
    coords, species = next(iter(dataloaded))
    #b, l, c = coords.size()
    #species = torch.LongTensor(torch.tensor([[6] * l]).repeat(b, 1))
    #print(species, next(iter(dataloaded)))
    #print(model.pre_block([species, next(iter(dataloaded))]) )
    print(model([species, coords]), species.shape)

