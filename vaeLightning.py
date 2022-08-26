import torch, torchani
import transformers as tfm
import dataloader as dl 
# import physnet_block as pb
import pytorch_lightning as pl
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
import vae as VAE
import argparse
from typing import *

pl.seed_everything(42)

logging.basicConfig()
logger = logging.getLogger("model.py logger")
logger.setLevel(logging.DEBUG)

warnings.simplefilter("ignore")
#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Model(pl.LightningModule):
    def __init__(self, args, **kwargs):
        super().__init__()
        
        self.args = args
        model_configs = kwargs.get("model_configs", None)
        self.model_block = VAE.VAE(args, **model_configs)     
        self.beta = args.beta
        self.data_mean = None
        self.data_std = None
        self.loader_length = None
        
        wandb.init(project="VAEGAN_MD", entity="hyunp2", name=args.name)
        wandb.watch(self.model_block, log="all")

#         latent_dim = self.model_block.decoder.hidden_dims[0] #2, 3, etc.
#         self.column_names = column_names = [f"z{i}" for i in range(latent_dim)]
      
    def forward(self, coords):
        #WIP: add atom/res types!
        z, mu, logstd, x = self.model_block(coords) #x: reconstructed 
        return z, mu, logstd, x

    def _shared_step(self, batch, batch_idx, return_metadata=False):
        coords = batch
        z, mu, logstd, x = self(coords) #coords: BL3 -> z: (B,latdim)
        mse, kl = self.model_block.losses(coords, z, mu, logstd, x, self.args.beta) #beta-VAE
        if not return_metadata:
            return mse, kl
        else:
            return mse, kl, z, mu, logstd, x
        
    def on_train_epoch_start(self, ) -> None:
        print(f"Current epoch is {self.current_epoch}!")
        
    def training_step(self, batch, batch_idx):
        mse, kl = self._shared_step(batch, batch_idx)
        wandb.log({
                   'train_kl': kl.item(),
                   'train_mse': mse.item(),
                   })
        loss = (mse - kl)
        self.log("train_loss", loss, prog_bar=True)
        return {"loss": loss, "train_kl": kl, "train_mse": mse}

    def training_epoch_end(self, training_step_outputs):
#         if not self.trainer.sanity_checking:
        epoch_train_loss = torch.stack([x["loss"] for x in training_step_outputs]).mean()
        epoch_train_mse = torch.stack([x["train_mse"] for x in training_step_outputs]).mean()
        epoch_train_kl = torch.stack([x["train_kl"] for x in training_step_outputs]).mean()
        self.log("epoch_train_loss", epoch_train_loss)
        wandb.log({'epoch': self.current_epoch,
                   'epoch_lr': self.trainer.optimizers[0].param_groups[0]["lr"],
                   'epoch_train_loss': epoch_train_loss,
                   'epoch_train_mse': epoch_train_mse,
                   'epoch_train_kl': epoch_train_kl,})

#     @staticmethod
#     def plot_manifold(args: argparse.ArgumentParser, mus: Union[np.ndarray, torch.Tensor], logstds: Union[np.ndarray, torch.Tensor], title: str, nologging=False):
#         #WIP for PCA or UMAP or MDS
#         #summary is 
#         import sklearn.manifold
#         import plotly.express as px
#         from umap import UMAP
#         import scipy.stats
# #         proj = sklearn.manifold.TSNE(2)
#         proj = UMAP(random_state=42)
#         mus_proj = proj.fit_transform(mus) #(B,2) of tsne
#         path_to_plotly_html = os.path.join(args.save_data_directory, "plotly_figure.html")
#         dist = scipy.stats.multivariate_normal(np.zeros(mus.shape[1]), 1)
#         fig = px.scatter(x=mus_proj[:,0], y=mus_proj[:,1], color=dist.pdf(mus).reshape(-1,))
#         fig.write_html(path_to_plotly_html, auto_play = False)
#         if not nologging:
#         table = wandb.Table(columns = ["plotly_figure"])
#         table.add_data(wandb.Html( open(path_to_plotly_html) ))
#         wandb.log({f"{proj.__class__.__name__} Plot {title}": table}) #Avoids overlap!
#         return proj #Fitted 

    @staticmethod
    def plot_manifold(args: argparse.ArgumentParser, mus: Union[np.ndarray, torch.Tensor], logstds: Union[np.ndarray, torch.Tensor], title: str):
        #WIP for PCA or UMAP or MDS
        #summary is 
        import plotly.express as px
        import scipy.stats
        
        assert len(mus.shape) == 2 and len(logstds.shape) == 2
        path_to_plotly_html = os.path.join(args.save_data_directory, "plotly_figure.html")
        dist = scipy.stats.multivariate_normal(np.zeros(mus.shape[1]), 1)
        table = wandb.Table(columns = ["plotly_figure"])
        colors = [mus, logstds]
        for i, c in enumerate(colors):
            if i == 0:
                fig = px.scatter(x=mus[:,0], y=mus[:,1], color=dist.pdf(c).reshape(-1,)) 
            elif i == 1:
                fig = px.scatter(x=mus[:,0], y=mus[:,1], color=np.exp(c.sum(axis=-1)).reshape(-1,)) 
            fig.write_html(path_to_plotly_html, auto_play = False)
            table.add_data(wandb.Html( open(path_to_plotly_html) ))
#         wandb.log({f"{proj.__class__.__name__} Plot {title}": table}) #Avoids overlap!
        wandb.log({f"Latent Plot {title}": table})
#         return proj #Fitted 

    @staticmethod
    def plot_manifold_with_colors(args: argparse.ArgumentParser, mus: Union[np.ndarray, torch.Tensor], logstds: Union[np.ndarray, torch.Tensor], title: str, colors: Union[np.ndarray, torch.Tensor]):
        #WIP for PCA or UMAP or MDS
        #summary is 
        import plotly.express as px
        import scipy.stats
        
        assert len(mus.shape) == 2 and len(logstds.shape) == 2
        path_to_plotly_html = os.path.join(args.save_data_directory, "plotly_figure.html")
#         dist = scipy.stats.multivariate_normal(np.zeros(mus.shape[1]), 1)
        table = wandb.Table(columns = ["plotly_figure"])
        fig = px.scatter(x=mus[:,0], y=mus[:,1], color=colors.reshape(-1,)) 
        fig.write_html(path_to_plotly_html, auto_play = False)
        table.add_data(wandb.Html( open(path_to_plotly_html) ))
        wandb.log({f"Latent Plot with Interps": table})

    def on_validation_epoch_start(self, ) -> None:
#         self.wandb_table = wandb.Table(columns=self.column_names)
#         self.df = []
#         print("Validation starts...")
        pass
        
    def validation_step(self, batch, batch_idx):
        mse, kl, z, mu, logstd, x = self._shared_step(batch, batch_idx, return_metadata=True)
        wandb.log({
                   'val_kl': kl.item(),
                   'val_mse': mse.item(),
                   })
        loss = (mse - kl)
        self.log("val_loss", loss, prog_bar=True)
        return {"val_loss": loss, "val_kl": kl, "val_mse": mse, "mu": mu, "logstd": logstd}

    def validation_epoch_end(self, validation_step_outputs):
#         if not self.trainer.sanity_checking:
        epoch_val_loss = torch.stack([x["val_loss"] for x in validation_step_outputs]).mean()
        epoch_val_mse = torch.stack([x["val_mse"] for x in validation_step_outputs]).mean()
        epoch_val_kl = torch.stack([x["val_kl"] for x in validation_step_outputs]).mean()
        mus = torch.cat([x["mu"] for x in validation_step_outputs], dim=0) #(b,dim) -> (B,dim)
        logstds = torch.cat([x["logstd"] for x in validation_step_outputs], dim=0) #(b,dim) -> (B,dim)
        self.log("epoch_val_loss", epoch_val_loss)
        wandb.log({
                   'epoch_val_loss': epoch_val_loss,
                   'epoch_val_mse': epoch_val_mse,
                   'epoch_val_kl': epoch_val_kl,
        })
        if self.current_epoch % 10 == 0:
            #WIP: Change modulus!
            print(mus.shape, logstds.shape)
            self.plot_manifold(self.args, mus.detach().cpu().numpy(), logstds.detach().cpu().numpy(), self.current_epoch)
#         df = torch.cat(self.df) #(MultiB, latent_dim)
#         self.wandb_table.add_data(*df.T)
#         wandb.run.log({f"Epoch {self.current_epoch} Valid Latent Representation" : wandb.plot.scatter(self.wandb_table,
#                             *self.column_names)})
    
    
    def on_test_epoch_start(self, ) -> None:
        print("Testing starts...")
    
    def test_step(self, batch, batch_idx):
        mse, kl, z, mu, logstd, x = self._shared_step(batch, batch_idx, return_metadata=True)
        wandb.log({
                   'test_kl': kl.item(),
                   'test_mse': mse.item(),
                   })
        loss = (mse - kl)
        self.log("test_loss", loss, prog_bar=True)
        return {"test_loss": loss, "test_kl": kl, "test_mse": mse, "mu": mu, "logstd": logstd}

    def test_epoch_end(self, test_step_outputs):
#         if not self.trainer.sanity_checking:
        epoch_test_loss = torch.stack([x["test_loss"] for x in test_step_outputs]).mean()
        epoch_test_mse = torch.stack([x["test_mse"] for x in test_step_outputs]).mean()
        epoch_test_kl = torch.stack([x["test_kl"] for x in test_step_outputs]).mean()
        mus = torch.cat([x["mu"] for x in test_step_outputs], dim=0) #(b,dim) -> (B,dim)
        logstds = torch.cat([x["logstd"] for x in test_step_outputs], dim=0) #(b,dim) -> (B,dim)
        self.log("epoch_test_loss", epoch_test_loss)
        wandb.log({
                   'epoch_test_loss': epoch_test_loss,
                   'epoch_test_mse': epoch_test_mse,
                   'epoch_test_kl': epoch_test_kl,
        })
        self.plot_manifold(self.args, mus.detach().cpu().numpy(), logstds.detach().cpu().numpy(), self.current_epoch)

#     def on_predict_epoch_start(self, ):
#         #Called per EPOCH!
#         self.wandb_table = wandb.Table(columns=self.column_names)
#         self.df = []

#     def predict_step(self, batch, batch_idx, dataloader_idx=0):
#         generate_molecules
        
#     def on_predict_epoch_end(self, ):
#         #Called per EPOCH!
#         df = torch.cat(self.df) #(MultiB, latent_dim)
#         self.wandb_table.add_data(*df.T)
#         wandb.run.log({"Pred Latent Representation" : wandb.plot.scatter(self.wandb_table,
#                             *self.column_names)})

    def configure_optimizers(self):
        optimizer = tfm.AdamW(self.model_block.parameters(), lr=self.args.lr)
        total_training_steps = self.loader_length * self.args.num_epochs
        warmup_steps = total_training_steps // self.args.warm_up_split
        if self.args.scheduler == "cosine":
            scheduler = tfm.get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps = warmup_steps, num_training_steps=total_training_steps, num_cycles = 1, last_epoch = -1)
        elif self.args.scheduler == "linear":
            scheduler = tfm.get_linear_schedule_with_warmup(optimizer, num_warmup_steps = warmup_steps, num_training_steps=total_training_steps, last_epoch = -1)
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1} #Every step/epoch with Frequency 1etc by monitoring val_loss if needed
        return [optimizer], [scheduler]


    def lerp(self, start: "1D tensor of features", end: "1D tensor of features", t: "1D tensor of weights"=1):
        outs = start + (end - start) * t.view(-1,1).to(self.device)
        return outs #(num_interpolations, dim)
        
    def _geometric_slerp(start, end, t):
        #https://github.com/scipy/scipy/blob/v1.9.0/scipy/spatial/_geometric_slerp.py#L35-L238:~:text=def%20geometric_slerp(,.ndarray%3A
        
        #WIP!
        
        # create an orthogonal basis using QR decomposition
        # One data point and interpolated!
        start = start.view(1,-1) 
        end = end.view(1,-1)
        basis = torch.cat([start,end], dim=0) #shape: (2,dim) #np.vstack([start, end])
        Q, R = torch.qr(basis.t()) ###SAME as reduced option of np.linalg.qr;;;; transpose -> (dim,2) --> BREAKS down into: (dim,k) and (k,dim)
        signs = 2 * (torch.diag(R) >= 0) - 1
        Q = Q.T * signs.T[:, None]
        R = R.T * signs.T[:, None]

        # calculate the angle between `start` and `end`
        c = start.view(-1,).dot(end.view(-1,))
        s = torch.linalg.det(R)
        omega = torch.atan2(s, c)

        # interpolate
        start, end = Q
        s = torch.sin(t * omega)
        c = torch.cos(t * omega)
        return start * c[:, None] + end * s[:, None]
    
    def slerp(self, t, v0, v1, DOT_THRESHOLD=0.9995):
        '''
        https://github.com/PDillis/stylegan2-fun/blob/master/run_generator.py
        Spherical linear interpolation
        Args:
            t (float/np.ndarray): Float value between 0.0 and 1.0
            v0 (np.ndarray): Starting vector
            v1 (np.ndarray): Final vector
            DOT_THRESHOLD (float): Threshold for considering the two vectors as
                                   colineal. Not recommended to alter this.
        Returns:
            v2 (np.ndarray): Interpolation vector between v0 and v1
        '''
        # Copy the vectors to reuse them later
        v0_copy = np.copy(v0)
        v1_copy = np.copy(v1)
        # Normalize the vectors to get the directions and angles
        v0 = v0 / np.linalg.norm(v0)
        v1 = v1 / np.linalg.norm(v1)
        # Dot product with the normalized vectors (can't use np.dot in W)
        dot = np.sum(v0 * v1)
        # If absolute value of dot product is almost 1, vectors are ~colineal, so use lerp
        if np.abs(dot) > DOT_THRESHOLD:
            return lerp(t, v0_copy, v1_copy)
        # Calculate initial angle between v0 and v1
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        # Angle at timestep t
        theta_t = theta_0 * t
        sin_theta_t = np.sin(theta_t)
        # Finish the slerp algorithm
        s0 = np.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        v2 = s0 * v0_copy + s1 * v1_copy
        return v2
    
    def generate_molecules(self, original: "test loader original coordinates (BL3)", start_idx: "starting point index, integer", end_idx: "end point index, integer", num_interps: "num of interpolations points") -> "Original && Recon_from_original && Lerp":
        """MOVE to pl.Callback!
        WIP: Add Plotly functions!"""
        original = original.to(self.device)
        z, mu, logstd, x = self(original) #"latend_coordinates of (B,latent_dim)"
        inputs = mu[start_idx] # dim,
        outputs = mu[end_idx] # dim,
        lerps = self.lerp(start=inputs, end=outputs, t=torch.linspace(0, 1, num_interps)[1:-1]) #(Num_interpolations by latent_dim)
        
        mean = self.data_mean #make sure dmo is saved as an argparse
        std = self.data_std
        unnormalize = dl.ProteinDataset.unnormalize #static method

        original_unscaled = unnormalize(original, mean=mean, std=std)
        recon_scaled = self.model_block.decoder(mu) #BL3, (scaled coord)
        recon  = unnormalize(recon_scaled, mean=mean, std=std) #BL3, test_loader latent to reconstruction (raw coord)
        lerps_recon_scaled = self.model_block.decoder(lerps.to(mu)) #BL3, lerped to reconstruction (scaled coord)
        lerps_recon  = unnormalize(lerps_recon_scaled, mean=mean, std=std) #BL3, test_loader latent to reconstruction (raw coord)
        print(original_unscaled, recon)
        colors = torch.cat([torch.LongTensor([i*10] * tensor.size(0)) for i, tensor in enumerate([original, recon, lerps_recon]) ], dim=0).detach().cpu().numpy()
        traj_cats = torch.cat([original, recon, lerps_recon], dim=0) #.detach().cpu().numpy() #BBB,L,3
        _, mus, logstds, _ = self(traj_cats)
        self.plot_manifold_with_colors(self.args, mus.detach().cpu().numpy(), logstds.detach().cpu().numpy(), None, colors)

#         psf = self.args.psf_file
        pdb = os.path.join(self.args.load_data_directory, os.path.splitext(self.args.pdb_file)[0] + "_reduced.pdb") #string
        atom_selection = self.args.atom_selection
    
        u = mda.Universe(pdb) #universe
        u.load_new(original_unscaled.detach().cpu().numpy()) #overwrite coords
        mda_traj_name = os.path.join(self.args.save_data_directory, self.args.name + "_test.dcd") if self.args.name is not None else os.path.join(self.args.save_data_directory, os.path.splitext(self.args.pdb_file)[0] + "_reduced" + "_test.dcd")
        with mda.Writer(mda_traj_name, u.atoms.n_atoms) as w:
            for ts in u.trajectory:
                w.write(u.atoms)   
    
        u = mda.Universe(pdb) #universe
        u.load_new(recon.detach().cpu().numpy()) #overwrite coords
        mda_traj_name = os.path.join(self.args.save_data_directory, self.args.name + "_recon.dcd") if self.args.name is not None else os.path.join(self.args.save_data_directory, os.path.splitext(self.args.pdb_file)[0] + "_reduced" + "_recon.dcd")
        with mda.Writer(mda_traj_name, u.atoms.n_atoms) as w:
            for ts in u.trajectory:
                w.write(u.atoms)   
                
        u = mda.Universe(pdb) #universe
        u.load_new(lerps_recon.detach().cpu().numpy()) #overwrite coords
        mda_traj_name = os.path.join(self.args.save_data_directory, self.args.name + "_lerps.dcd") if self.args.name is not None else os.path.join(self.args.save_data_directory, os.path.splitext(self.args.pdb_file)[0] + "_reduced" + "_lerps.dcd")
        with mda.Writer(mda_traj_name, u.atoms.n_atoms) as w:
            for ts in u.trajectory:
                w.write(u.atoms)   
        
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

