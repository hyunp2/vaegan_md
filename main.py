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

#/Scr/hyunpark/anaconda3/envs/deeplearning

def get_args():
    parser = argparse.ArgumentParser(description='Training')

    #Model related
    parser.add_argument('--load-model-directory', "-dirm", type=str, default="/Scr/hyunpark/Monster/vaegan/output", help='This is where model is/will be located...')  
    parser.add_argument('--load-model-checkpoint', "-ckpt", type=str, default=None, help='which checkpoint...')  
    parser.add_argument('--model', type=str, default='physnet', choices=["physnet"], help='Which model to train')
    parser.add_argument('--loss_type', type=str, default="total", choices=["boltzmann", "contrastive", "total"], help='Loss functions')
    parser.add_argument('--save-interval', type=int, default=1, help='Save interval, one save per n epochs (default: 1)')
    parser.add_argument('--model-config', "-config", type=str, default=None, help='Energy model configuration')

    #Molecule (Dataloader) related
    parser.add_argument('--load-data-directory', "-dird", default="/Scr/hyunpark/Monster/PL_REDO/data", help='This is where data is located...')  
    parser.add_argument('--save-data-directory', "-sird", default="/Scr/hyunpark/Monster/PL_REDO/data", help='This is where data is located...')  
    parser.add_argument('--molecule', type=str, default="alanine-dipeptide", help='Which molecule to analyze')
    parser.add_argument('--atom-selection', '-asel', type=str, default="all", help='MDAnalysis atom selection')
    parser.add_argument('--psf-file', '-psf', type=str, default=None, help='MDAnalysis PSF')
    parser.add_argument('--pdb-file', '-pdb', type=str, default=None, help='MDAnalysis PDB')
    parser.add_argument('--trajectory-files', '-traj', type=str, nargs='+', default=None, help='MDAnalysis Trajectories')
    parser.add_argument('--split-portion', '-spl', type=int, default=80, help='Torch dataloader and Pytorch lightning split of batches')

    #Optimizer related
    parser.add_argument('--num-epochs', default=60, type=int, help='number of epochs')
    parser.add_argument('--batch-size', '-b', default=2048, type=int, help='batch size')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--ngpus', type=int, default=-1, help='Number of GPUs, -1 use all available. Use CUDA_VISIBLE_DEVICES=1, to decide gpus')
    parser.add_argument('--num-nodes', type=int, default=1, help='Number of nodes')
    parser.add_argument('--warm-up-split', type=int, default=5, help='warmup times')
    parser.add_argument('--scheduler', type=str, default="cosine", help='scheduler type')

    #Misc.
    parser.add_argument('--precision', type=int, default=32, choices=[16, 32], help='Floating point precision')
    parser.add_argument('--distributed-backend', default='ddp', help='Distributed backend: dp, ddp, ddp2')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers for data prefetch')

    args = parser.parse_args()
    return args

def _main():
    args = get_args()

    pl.seed_everything(args.seed)

    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = Model.ProtBertClassifier.load_from_checkpoint( os.path.join(args.load_model_directory, args.load_model_checkpoint), hparam=args, strict=True ) if args.load_model_checkpoint else Model(args)

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
    filename="{epoch}-{train_loss_mean:.2f}-{val_loss_mean:.2f}",
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

    trainer.fit(model) #New API!
    
def _test(args: argparse.ArgumentParser):
#     hparams = get_args()

    pl.seed_everything(args.seed)

    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = Model.ProtBertClassifier.load_from_checkpoint( os.path.join(args.load_model_directory, args.load_model_checkpoint), hparam=args, strict=True )
    print("PASS")
    
    if args.load_model_checkpoint:
        resume_ckpt = os.path.join(args.load_model_directory, args.load_model_checkpoint)
    else:
        resume_ckpt = None
        
    if args.strategy in ["none", None]:
        args.strategy = None
        
    csv_logger = pl.loggers.CSVLogger(save_dir=args.load_model_directory)

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
#     python -m train --json_directory /Scr/hyunpark/DL_Sequence_Collab/ProtLIpInt/ --save_to_file data_compiled.pickle --train_mode train   
    elif args.train_mode in ["test"]:
        _test()
#     python -m train --json_directory /Scr/hyunpark/DL_Sequence_Collab/ProtLIpInt/ --save_to_file data_compiled.pickle --load_model_checkpoint epoch=59-train_loss_mean=0.08-val_loss_mean=0.10.ckpt --train_mode test  

    elif args.train_mode in ["pred"]:
        _test(args)
#     python -m train --json_directory /Scr/hyunpark/DL_Sequence_Collab/ProtLIpInt/ --save_to_file data_compiled.pickle --load_model_checkpoint epoch=59-train_loss_mean=0.08-val_loss_mean=0.10.ckpt --train_mode pred  


    

if __name__ == "__main__":
    main()
