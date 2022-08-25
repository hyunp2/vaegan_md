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
from vaeLightning import Model
import dataloader as dl 

#/Scr/hyunpark/anaconda3/envs/deeplearning

def get_args():
    parser = argparse.ArgumentParser(description='Training')

    #Model related
    parser.add_argument('--load_model_directory', "-dirm", type=str, default="/Scr/hyunpark/Monster/vaegan_md/output", help='Model ROOT directory...')  
    parser.add_argument('--load_model_checkpoint', "-ckpt", type=str, default=None, help='Find NAME of a CHECKPOINT')  
    parser.add_argument('--name', type=str, default=None, help='Name for Wandb and GENERATED data!')  

    #Molecule (Dataloader) related
    parser.add_argument('--load_data_directory', "-dird", default="/Scr/hyunpark/Monster/vaegan_md/data", help='Locate ORIGINAL data')  
    parser.add_argument('--save_data_directory', "-sird", default="/Scr/hyunpark/Monster/vaegan_md/generated_data", help='Save GENERATED data')  
    parser.add_argument('--psf_file', '-psf', type=str, default=None, help='MDAnalysis PSF')
    parser.add_argument('--pdb_file', '-pdb', type=str, default=None, help='MDAnalysis PDB')
    parser.add_argument('--trajectory_files', '-traj', type=str, nargs='*', default=None, help='MDAnalysis Trajectories')
    parser.add_argument('--molecule', type=str, default="alanine-dipeptide", help='Which molecule to analyze')
    parser.add_argument('--atom_selection', '-asel', type=str, default="all", help='MDAnalysis atom selection')
    parser.add_argument('--split_portion', '-spl', type=int, default=80, help='Torch dataloader and Pytorch lightning split of batches')

    #Optimizer related
    parser.add_argument('--num_epochs', default=100, type=int, help='number of epochs')
    parser.add_argument('--batch_size', '-b', default=128, type=int, help='batch size')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--ngpus', type=int, default=-1, help='Number of GPUs, -1 use all available. Use CUDA_VISIBLE_DEVICES=1, to decide gpus')
    parser.add_argument('--num_nodes', type=int, default=1, help='Number of nodes')
    parser.add_argument('--warm_up_split', type=int, default=5, help='warmup times')
    parser.add_argument('--scheduler', type=str, default="cosine", help='scheduler type')
    parser.add_argument('--max_epochs', default=60, type=int, help='number of epochs max')
    parser.add_argument('--min_epochs', default=1, type=int, help='number of epochs min')
    parser.add_argument('--precision', type=int, default=32, choices=[16, 32], help='Floating point precision')
    parser.add_argument('--monitor', type=str, default="epoch_val_loss", help='metric to watch')
    parser.add_argument('--loss', '-l', type=str, default="classification", choices=['classification', 'contrastive', 'ner'], help='loss for training')
    parser.add_argument('--save_top_k', type=int, default="5", help='num of models to save')
    parser.add_argument('--patience', type=int, default=10, help='patience for stopping')
    parser.add_argument('--metric_mode', type=str, default="min", help='mode of monitor')
    parser.add_argument('--amp_backend', type=str, default="native", help='Torch vs NVIDIA AMP')
    parser.add_argument('--sanity_checks', '-sc', type=int, default=2, help='Num sanity checks..')
    parser.add_argument('--accelerator', "-accl", type=str, default="gpu", help='accelerator type', choices=["cpu","gpu","tpu"])
    parser.add_argument('--strategy', "-st", default="ddp", help='accelerator type', choices=["ddp_spawn","ddp","dp","ddp2","horovod","none"])
    
    #Misc.
    parser.add_argument('--distributed_backend', default='ddp', help='Distributed backend: dp, ddp, ddp2')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data prefetch')
    parser.add_argument('--train_mode', type=str, default="train", choices=["train","test","pred"])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--beta', type=float, default=1., help="beta-VAE")

    args = parser.parse_args()
    return args

def _main():
    args = get_args()
    
    pl.seed_everything(args.seed)

    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    atom_selection = args.atom_selection
    pdb = os.path.join(args.load_data_directory, args.pdb_file) #string
    psf = os.path.join(args.load_data_directory, args.psf_file) #string
    prot_ref = mda.Universe(psf, pdb)
    pos = prot_ref.atoms.select_atoms(atom_selection).positions #L,3
    unrolled_dim = pos.shape[0] * pos.shape[1]
    
    model_configs = dict(hidden_dims_enc=[1500, 750, 400, 200, 200],
                         hidden_dims_dec=[100, 200, 400, 750, 1500],
                         unrolled_dim=unrolled_dim)
    model = Model.Model.load_from_checkpoint( os.path.join(args.load_model_directory, args.load_model_checkpoint), args=args, model_configs=model_configs, strict=True ) if args.load_model_checkpoint else Model(args=args, model_configs=model_configs)

    # ------------------------
    # 2 INIT EARLY STOPPING
    # ------------------------
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor=args.monitor,
        min_delta=0.0,
        patience=args.patience,
        verbose=True,
        mode=args.metric_mode,
    )

    # --------------------------------
    # 3 INIT MODEL CHECKPOINT CALLBACK
    #  -------------------------------
    # initialize Model Checkpoint Saver
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
    filename="{epoch}-{epoch_val_loss:.2f}",
    save_top_k=args.save_top_k,
    verbose=True,
    monitor=args.monitor,
    every_n_epochs=1,
    mode=args.metric_mode,
    dirpath=args.load_model_directory,
    )

    # --------------------------------
    # 4 INIT SWA CALLBACK
    #  -------------------------------
    # Stochastic Weight Averaging
    swa_callback = pl.callbacks.StochasticWeightAveraging(swa_epoch_start=0.8, swa_lrs=None, annealing_epochs=10, annealing_strategy='cos', avg_fn=None)

    # --------------------------------
    # 5 INIT SWA CALLBACK
    #  -------------------------------
    # Stochastic Weight Averaging
#     rsummary_callback = pl.callbacks.RichModelSummary() #Not in this PL version

    # --------------------------------
    # 6 INIT MISC CALLBACK
    #  -------------------------------
    # MISC
#     progbar_callback = pl.callbacks.ProgressBar()
    timer_callback = pl.callbacks.Timer()
    tqdmbar_callback = pl.callbacks.TQDMProgressBar()
    
    # ------------------------
    # N INIT TRAINER
    # ------------------------
    csv_logger = pl.loggers.CSVLogger(save_dir=args.load_model_directory)
#     plugins = DDPPlugin(find_unused_parameters=False) if hparams.accelerator == "ddp" else None
    
    # ------------------------
    # MISC.
    # ------------------------
    if args.load_model_checkpoint:
        resume_ckpt = os.path.join(args.load_model_directory, args.load_model_checkpoint)
    else:
        resume_ckpt = None
        
    if args.strategy in ["none", None]:
        args.strategy = None
    
    datamodule = dl.DataModule(args)
    datamodule.setup()
    train_dataloaders, val_dataloaders = datamodule.train_dataloader(), datamodule.val_dataloader()
    [setattr(model, key, val) for key, val in zip(["data_mean", "data_std", "loader_length"], [datamodule.mean, datamodule.std, datamodule.trajectory.size(0) ])] #set mean and std
    print("Model's dataset mean and std are set:", model.data_mean, " and ", model.data_std)

    trainer = pl.Trainer(
        logger=[csv_logger],
        max_epochs=args.max_epochs,
        min_epochs=args.min_epochs,
        callbacks = [early_stop_callback, checkpoint_callback, swa_callback, tqdmbar_callback, timer_callback],
        precision=args.precision,
        amp_backend=args.amp_backend,
        deterministic=False,
        default_root_dir=args.load_model_directory,
        num_sanity_val_steps = args.sanity_checks,
        log_every_n_steps=4,
        gradient_clip_algorithm="norm",
        gradient_clip_val=1.,
        devices=args.ngpus,
        strategy=args.strategy,
        accelerator=args.accelerator,
        auto_select_gpus=True,
    )

    trainer.fit(model, train_dataloaders, val_dataloaders) #New API!
    
def _test(args: argparse.ArgumentParser):
#     hparams = get_args()

    pl.seed_everything(args.seed)

    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    atom_selection = args.atom_selection
    pdb = os.path.join(args.load_data_directory, args.pdb_file) #string
    psf = os.path.join(args.load_data_directory, args.psf_file) #string
    prot_ref = mda.Universe(psf, pdb)
    pos = prot_ref.atoms.select_atoms(atom_selection).positions #L,3
    unrolled_dim = pos.shape[0] * pos.shape[1]
    
    model_configs = dict(hidden_dims_enc=[1500, 750, 400, 200, 200],
                         hidden_dims_dec=[100, 200, 400, 750, 1500],
                         unrolled_dim=unrolled_dim)
    model = Model.Model.load_from_checkpoint( os.path.join(args.load_model_directory, args.load_model_checkpoint), args=args, model_configs=model_configs, strict=True ) if args.load_model_checkpoint else Model(args=args, model_configs=model_configs)
    
    if args.load_model_checkpoint:
        resume_ckpt = os.path.join(args.load_model_directory, args.load_model_checkpoint)
    else:
        resume_ckpt = None
        
    if args.strategy in ["none", None]:
        args.strategy = None
        
    csv_logger = pl.loggers.CSVLogger(save_dir=args.load_model_directory)

    datamodule = dl.DataModule(args)
    datamodule.setup()
    test_dataloaders = datamodule.test_dataloader()
    [setattr(model, key, val) for key, val in zip(["data_mean", "data_std", "loader_length"], [datamodule.mean, datamodule.std, datamodule.trajectory.size(0) ])] #set mean and std
    print("Model's dataset mean and std are set:", model.data_mean, " and ", model.data_std)
    
    trainer = pl.Trainer(
        logger=[csv_logger],
        max_epochs=args.max_epochs,
        min_epochs=args.min_epochs,
        precision=args.precision,
        amp_backend=args.amp_backend,
        deterministic=False,
        default_root_dir=args.load_model_directory,
        num_sanity_val_steps = args.sanity_checks,
        log_every_n_steps=4,
        gradient_clip_algorithm="norm",
        gradient_clip_val=1.,
        devices=args.ngpus,
        strategy=args.strategy,
        accelerator=args.accelerator,
        auto_select_gpus=True,
    )
    if args.train_mode in ["test"]:
        trainer.test(model) #New API!
    elif args.train_mode in ["pred"]:
        test_dataloader = model.test_dataloader()
        trainer.predict(model, dataloaders=test_dataloader)

if __name__ == "__main__":
    args = get_args()
    print(args.train_mode)
    
    if args.train_mode in ["train"]:
        _main()
    elif args.train_mode in ["test"]:
        _test()

    elif args.train_mode in ["pred"]:
        _test(args)

if __name__ == "__main__":
    main()
#     python -m main --psf_file 3f48final.psf --pdb_file 3f48finaleqnbfix.pdb --trajectory_files force5tm18_afterreleaseeq_nbfix.dcd --strategy none --batch_size 16
