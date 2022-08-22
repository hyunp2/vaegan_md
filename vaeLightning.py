import torch, torchani
import transformers as tfm
import dataloader as dl 
# import physnet_block as pb
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
import vae as VAE

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
        self.model_block = VAE.VAE(**energy_configs)     
        wandb.init(project="VAE-GAN", entity="hyunp2")
        latent_dim = self.model_block.decoder.hidden_dims[0] #2, 3, etc.
        self.column_names = column_names = [f"z{i}" for i in range(latent_dim)]
        wandb.watch(self.model_block, log="all")
      
    def forward(self, inputs):
        coords = inputs #Input of original coords
        z, mu, logstd, x = self.model_block(coords) #reconstructed 

    def training_step(self, batch, batch_idx):
        loss, kl, mse, rmsd, tm, mat = self._shared_step(batch, batch_idx)
        wandb.log({'train_loss': loss.item(),
                   'train_kl': kl.item(),
                   'train_mse': mse.item(),
                   'train_rmsd': rmsd.item(),
                   'train_tm': tm.item(),
                   'train_mat': mat.item()})
        return loss

    def training_epoch_end(self, training_step_outputs):
        if not self.trainer.running_sanity_check:
            self.log("epoch_train_loss", torch.stack(training_step_outputs).mean(), prog_bar=True)
            wandb.log({'epoch': self.current_epoch,
                       'epoch_lr': self.trainer.optimizers[0].param_groups[0]["lr"],
                       'epoch_train_loss': torch.stack(training_step_outputs).mean().item()})

    def on_validation_start(self, ):
        #Called per EPOCH!
        self.wandb_table = wandb.Table(columns=self.column_names)
        self.df = []
        
    def on_validation_end(self, ):
        #Called per EPOCH!
        df = torch.cat(self.df) #(MultiB, latent_dim)
        self.wandb_table.add_data(*df.T)
        wandb.run.log({f"Epoch {self.current_epoch} Valid Latent Representation" : wandb.plot.scatter(self.wandb_table,
                            *self.column_names)})

    def validation_step(self, batch, batch_idx):
        loss, kl, mse, rmsd, tm, mat = self._shared_step(batch, batch_idx, stage="valid")
        wandb.log({'val_loss': loss.item(),
                   'val_kl': kl.item(),
                   'val_mse': mse.item(),
                   'val_rmsd': rmsd.item(),
                   'val_tm': tm.item(),
                   'val_mat': mat.item()})
        return loss

    def validation_epoch_end(self, validation_step_outputs):
        if not self.trainer.running_sanity_check:
            self.log("epoch_val_loss", torch.stack(validation_step_outputs).mean(), prog_bar=True)
            wandb.log({
                       'epoch_val_loss': torch.stack(validation_step_outputs).mean().item()})

    def test_step(self, batch, batch_idx):
        loss, kl, mse, rmsd, tm, mat = self._shared_step(batch, batch_idx)
        wandb.log({'test_loss': loss.item(),
                   'test_kl': kl.item(),
                   'test_mse': mse.item(),
                   'test_rmsd': rmsd.item(),
                   'test_tm': tm.item(),
                   'test_mat': mat.item()})
        return loss

    def test_epoch_end(self, test_step_outputs):
        if not self.trainer.running_sanity_check:
            self.log("epoch_test_loss", torch.stack(test_step_outputs).mean(), prog_bar=True)
            wandb.log({
                       'epoch_test_loss': torch.stack(test_step_outputs).mean().item()})

    def _shared_step(self, batch, batch_idx, stage=None):
        coords = batch
        z, mu, logstd, x = self.model_block(coords)
        if stage in ["valid", "pred"]: self.df.append(mu) #list of (B,latent_dim)
        loss, kl, mse, rmsd, tm, mat = self.custom_loss(z, mu, logstd, x)
        return loss, kl, mse, rmsd, tm, mat 

    def on_predict_epoch_start(self, ):
        #Called per EPOCH!
        self.wandb_table = wandb.Table(columns=self.column_names)
        self.df = []
        
    def on_predict_end(self, ):
        #Called per EPOCH!
        df = torch.cat(self.df) #(MultiB, latent_dim)
        self.wandb_table.add_data(*df.T)
        wandb.run.log({"Pred Latent Representation" : wandb.plot.scatter(self.wandb_table,
                            *self.column_names)})

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        self._shared_step(batch, batch_idx, stage="pred")

    def configure_optimizers(self):
        optimizer = tfm.AdamW(self.model_block.parameters(), lr=self.args.lr)
        total_training_steps = self.loader_length * self.args.num_epochs
        warmup_steps = total_training_steps // self.args.warm_up_split
        if self.scheduler == "cosine":
            scheduler = tfm.get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps = warmup_steps, num_training_steps=total_training_steps, num_cycles = 1, last_epoch = -1)
        elif self.scheduler == "linear":
            scheduler = tfm.get_linear_schedule_with_warmup(optimizer, num_warmup_steps = warmup_steps, num_training_steps=total_training_steps, last_epoch = -1)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def custom_loss(self, z, mu, logstd, x, coeffs: List[int]):
        kl, mse, rmsd, tm, mat = VAE.VAE.losses(z, mu, logstd, x) #static method  
        assert len(coeffs) == 5 #same as number of losses
        loss = torch.sum(list(map(lambda coeff_, loss_: coeff_ * loss_, (*coeffs), (kl, mse, rmsd, tm, mat) )))
        return loss, kl, mse, rmsd, tm, mat   

    def lerp(self, start: "1D tensor of features", end: "1D tensor of features", t: "1D tensor of weights"=1):
        outs = start + (end - start) * t.view(-1,1).to(inputs)
        return outs

    def _geometric_slerp(start, end, t):
        #https://github.com/scipy/scipy/blob/v1.9.0/scipy/spatial/_geometric_slerp.py#L35-L238:~:text=def%20geometric_slerp(,.ndarray%3A
        
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
    
    def generate_molecules(self, original: "test loader original coordinates (BL3)", inps: "starting point index, integer", outs: "end point index, integer", interps: "num of interpolations points") -> "Original && Recon_from_original && Lerp":
        """MOVE to pl.Callback!"""
        original = original
        z, latent, logstd = self.model_block.encoder(original) #"latend_coordinates of (B,latent_dim)"
        inputs = latent[inps]
        outputs = latent[outs]
        lerps = self.lerp(inputs=inputs, outputs=outputs, interps=torch.linspace(0, 1, interps)[1:-1]) #(Num_interpolations by latent_dim)
        mean = self.args.dmo.dataset.mean #make sure dmo is saved as an argparse
        std = self.args.dmo.dataset.std
        unnormalize = dl.ProteinDataset.unnormalize #static method

        x = self.model_block.decoder(latent.to(mu))
        original_recon  = unnormalize(x, mean=mean, std=std) #BL3, test_loader latent to reconstruction (raw coord)
        x = self.model_block.decoder(lerps.to(mu)) #BL3, lerped to reconstruction (raw coord)
        lerp_recon  = unnormalize(x, mean=mean, std=std) #BL3, test_loader latent to reconstruction (raw coord)

        psf = self.args.psf_file
        pdb = self.args.pdb_file
        atom_selection = self.args.atom_selection
        
        universes = [mda.Universe(psf, pdb) for _ in range(3)]
        universe_atomgroups = list(map(lambda inp: getattr(inp, "select_atoms")(atom_selection), universes))

        fake_closed_ca = fake_closed.atoms.select_atoms(atom_selection)
        for idx, _ in enumerate(recon_):
        fake_closed_ca.atoms.positions = unnormalize(traj, mean, std)[idx].detach().cpu().numpy()
        fake_closed_ca.write(f"REAL_Train{idx}.pdb")
        fake_closed_ca.atoms.positions = RECON[idx].detach().cpu().numpy()
        fake_closed_ca.write(f"FAKE_Train{idx}.pdb")
        fake_closed_ca.atoms.positions = recon_[idx].detach().cpu().numpy()
        fake_closed_ca.write(f"fake_INTERP{idx}.pdb")
        fake_closed_ca.atoms.positions = recon2_[idx].detach().cpu().numpy()
        fake_closed_ca.write(f"fake_INTERP{idx}_2.pdb")
    plt.scatter(*mu[:102].T.detach().cpu().numpy(), c=np.arange(102)); plt.scatter(*mu[102:].T.detach().cpu().numpy(), c=np.arange(98)); plt.colorbar(); plt.scatter(*lerps.T.detach().cpu().numpy()); plt.scatter(*lerps2.T.detach().cpu().numpy());  
    plt.scatter(*(mu + logstd.exp()*torch.randn_like(logstd)*0.01).T.detach().cpu().numpy()); plt.show()
    


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

