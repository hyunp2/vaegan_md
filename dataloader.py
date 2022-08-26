import torch
import einops
from einops.layers.torch import Rearrange
import itertools
import numpy as np
import matplotlib.pyplot as plt
from curtsies import fmtfuncs as cf
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
# import main as Main
import pdb as PDB
from MDAnalysis.analysis.align import AlignTraj

__all__ = ["DataModule"]

def alignTrajectory(args):
    atom_selection = args.atom_selection
    assert args.pdb_file != None and args.psf_file != None, "both PDB and PSF must be provides..."
    #PDB.set_trace()
    print(args.load_data_directory, args.pdb_file, os.path.join(args.load_data_directory, args.pdb_file))
    pdb = os.path.join(args.load_data_directory, args.pdb_file) #string
    psf = os.path.join(args.load_data_directory, args.psf_file) #string
    traj = list(map(lambda inp: os.path.join(args.load_data_directory, inp), args.trajectory_files)) #string list
    prot_ref = mda.Universe(psf, pdb) #must not call PSF
    prot_traj = mda.Universe(psf, *traj) 
    print(prot_traj.atoms.positions.shape)
    AlignTraj(prot_traj, prot_ref, select=args.atom_selection, in_memory=True).run()
    print(prot_traj.atoms.positions.shape)
    
def extract_trajectory(args):
    atom_selection = args.atom_selection
    assert args.pdb_file != None and args.psf_file != None, "both PDB and PSF must be provides..."
    #PDB.set_trace()
    print(args.load_data_directory, args.pdb_file, os.path.join(args.load_data_directory, args.pdb_file))
    pdb = os.path.join(args.load_data_directory, args.pdb_file) #string
    psf = os.path.join(args.load_data_directory, args.psf_file) #string
    traj = list(map(lambda inp: os.path.join(args.load_data_directory, inp), args.trajectory_files)) #string list
    prot_ref = mda.Universe(pdb) #must not call PSF
    prot_traj = mda.Universe(psf, *traj) 
    prot_ref_ag = prot_ref.atoms.select_atoms(f"{atom_selection}")
    
    reduced_pdb_file = os.path.join(args.load_data_directory, os.path.splitext(args.pdb_file)[0] + "_reduced.pdb")
#     if not os.path.exists(reduced_pdb_file):
    prot_ref_ag.write(reduced_pdb_file) #write a reduced file; based on atom selection!
#     else:
#         pass
    reference = torch.from_numpy(prot_ref.atoms.select_atoms(f"{atom_selection}").positions)[None,:]
    coords = AnalysisFromFunction(lambda ag: ag.positions.copy(),
                                   prot_traj.atoms.select_atoms(f"{atom_selection}")).run().results['timeseries']
    trajectory = torch.from_numpy(coords) #Train
    assert reference.ndim == trajectory.ndim, "Must have 3 dimensions for both REF and TRAJ..."
    assert isinstance(reference, torch.Tensor) and isinstance(trajectory, torch.Tensor), "Both reference and trajectory should be torch tensors!"
    print(cf.on_yellow(f"Protesin has {trajectory.size(1)} atoms selected!"))
    return reference, trajectory #Both are atom_selection (i.e. potentially reduced!)

class ProteinDataset(torch.utils.data.Dataset):
    """Normalized dataset and reverse-normalization happens here..."""
    def __init__(self, dataset: List[Union[torch.Tensor, np.array]]):
        super().__init__()
        self.reference = dataset[0]
        self.trajectory = dataset[1]
        self.trajectory, self.mean, self.std = self.normalize(self.trajectory) #Raw (x,y,z) to Normalized (x,y,z)
        assert self.reference.ndim == 3 and self.trajectory.ndim == 3, "dimensions are incorrect..."
        
    def __len__(self):
        return len(self.trajectory) #train length...
    
    def __getitem__(self, idx):
        coords = self.trajectory[idx] #B, L, 3
        return coords
    
    def normalize(self, coords):
        coords = coords.view(coords.size(0), -1)
        mean = coords.mean(dim=0) #(B,C)
        std = coords.std(dim=0) #(B,C)
        coords_ = (coords - mean) / std
        coords_ = coords_.view(coords.size(0), coords.size(1)//3 ,3) #Back to original shape (B,L,3)
        return coords_, mean, std #coords_ is SCALED BL3 shape dataset!
    
    @staticmethod
    def unnormalize(coords, mean=None, std=None):
        assert mean != None and std != None, "Wrong arguments..."
        coords = coords.view(coords.size(0), -1)
        coords_ = (coords * std) + mean
        coords_ = coords_.view(coords.size(0), coords.size(1)//3 ,3) #Back to original shape (B,L,3)
        return coords_ #Reconstructed unscaled (i.e. raw) dataset (BL3)

class DataModule(pl.LightningDataModule):
    def __init__(self, args=None, **kwargs):
        super(DataModule, self).__init__()
        datasets = extract_trajectory(args) #tuple of reference and traj
        self.dataset = dataset = ProteinDataset(datasets)
        self.reference = dataset.reference #Reference data of (1,L,3)
        self.trajectory = dataset.trajectory #Trajectory (B,L,3)
        self.mean = dataset.mean
        self.std = dataset.std
        self.batch_size = args.batch_size
        split_portion = args.split_portion
        self.seed = kwargs.get("seed", 42)
        assert split_portion < 100 and split_portion > 0, "this parameter must be a positive number less than 100..."
        self.split_portion = (split_portion / 100) if split_portion > 1 else split_portion
        self.train_data_length = int(len(self.dataset) * self.split_portion)
        self.valid_data_length = int(len(self.dataset) * (1 - self.split_portion)/2 )
        self.train_val_dataset = self.trajectory[ : (self.train_data_length+self.valid_data_length) ]
        self.test_dataset = self.trajectory[(self.train_data_length+self.valid_data_length) : ]
        self.num_workers = args.num_workers

    #@pl.utilities.distributed.rank_zero_only
    def setup(self, stage=None):
        self.trainset, self.validset= torch.utils.data.random_split(self.train_val_dataset, [self.train_data_length, self.valid_data_length], generator=torch.Generator().manual_seed(self.seed)) 
        self.testset = self.test_dataset #Pristine Last frames in correct times...
        
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.trainset, shuffle=True, num_workers=self.num_workers, batch_size=self.batch_size, drop_last=False, pin_memory=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.validset, shuffle=False, num_workers=self.num_workers, batch_size=self.batch_size, drop_last=False, pin_memory=True)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.testset, shuffle=False, num_workers=self.num_workers, batch_size=self.batch_size, drop_last=False, pin_memory=True)

if __name__ == "__main__":
    from main import get_args
    args = get_args()
    alignTrajectory(args)
    
    
#     args = Main.get_args()
#     args.batch_size = 50
#     reference, trajectory = extract_trajectory(args)
#     dataset = ProteinDataset([reference, trajectory])
#     dmo = DataModule(dataset, args, seed=42)
#     dmo.prepare_data()
#     dmo.setup()
#     print(next(iter(dmo.train_dataloader())))
#     print((ProteinDataset.unnormalize(next(iter(dmo.train_dataloader())), dataset.mean, dataset.std)))
#     #python dataloader.py -psf reference_autopsf.psf -pdb reference_autopsf.pdb -traj adk.dcd

