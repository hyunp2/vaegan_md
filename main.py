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
    parser.add_argument('--load-model-directory', "-dirm", type=str, default="/Scr/hyunpark/Monster/PL_REDO/output", help='This is where model is/will be located...')  
    parser.add_argument('--load-model-checkpoint', "-ckpt", type=str, default=None, help='which checkpoint...')  
    parser.add_argument('--model', type=str, default='physnet', choices=["physnet"], help='Which model to train')
    parser.add_argument('--loss_type', type=str, default="total", choices=["boltzmann", "contrastive", "total"], help='Loss functions')
    parser.add_argument('--save-interval', type=int, default=1, help='Save interval, one save per n epochs (default: 1)')
    parser.add_argument('--model-config', "-config", type=str, default=None, help='Energy model configuration')

    #Molecule (Dataloader) related
    parser.add_argument('--load-data-directory', "-dird", default="/Scr/hyunpark/Monster/PL_REDO/data", help='This is where data is located...')  
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

def main():
    args = get_args()
    #Training logic will come soon...
    

if __name__ == "__main__":
    main()
