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

warnings.simplefilter("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_printoptions(precision=4)

###########################
##########Utils############
###########################

def kabsch_torch(X, Y, cpu=True):
    """ Kabsch alignment of X into Y. 
        Assumes X,Y are both (Dims x N_points). See below for wrapper.
    """
    device = X.device
    #  center X and Y to the origin
    X_ = X - X.mean(dim=-1, keepdim=True)
    Y_ = Y - Y.mean(dim=-1, keepdim=True)
    # calculate convariance matrix (for each prot in the batch)
    C = torch.matmul(X_, Y_.t()).detach()
    if cpu: 
        C = C.cpu()
    # Optimal rotation matrix via SVD
    if int(torch.__version__.split(".")[1]) < 8:
        # warning! int torch 1.<8 : W must be transposed
        V, S, W = torch.svd(C)
        W = W.t()
    else: 
        V, S, W = torch.linalg.svd(C)
    
    # determinant sign for direction correction
    d = (torch.det(V) * torch.det(W)) < 0.0
    if d:
        S[-1]    = S[-1] * (-1)
        V[:, -1] = V[:, -1] * (-1)
    # Create Rotation matrix U
    U = torch.matmul(V, W).to(device)
    # calculate rotations
    X_ = torch.matmul(X_.t(), U).t()
    # return centered and aligned
    return X_, Y_


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

###########################
##########VAEs#############
###########################
class MultiheadAttention_Residual(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.rolled_dim = kwargs.get("rolled_dim")
        self.mha_dimension = kwargs.get("mha_dimension", 1200)
        self.nheads = kwargs.get("nheads", 6)
        self.register_buffer("dim_per_head", torch.tensor(self.mha_dimension / self.nheads))
        assert self.mha_dimension  % self.nheads == 0, "Indivisible..."
        self.q = torch.nn.Linear(self.rolled_dim, self.mha_dimension) #temporarily change to lower dimension
        self.kv = torch.nn.Linear(self.rolled_dim, self.mha_dimension*2)
        self.ff = torch.nn.Sequential(torch.nn.Linear(self.mha_dimension, self.rolled_dim))
        self.dropout = torch.nn.Dropout(0.2)
        self.norm = torch.nn.LayerNorm(self.rolled_dim)
        #self.norm = torch.nn.LayerNorm(self.mha_dimension)
        self.ff2 = torch.nn.Sequential(*[torch.nn.Linear(self.rolled_dim, 4*self.rolled_dim), torch.nn.LeakyReLU(True), torch.nn.Linear(self.rolled_dim*4, self.rolled_dim)])
        #self.ff2 = torch.nn.Sequential(*[torch.nn.Linear(self.mha_dimension, 4*self.mha_dimension), torch.nn.LeakyReLU(True), torch.nn.Linear(self.mha_dimension*4, self.mha_dimension), torch.nn.LeakyReLU(True), torch.nn.Linear(self.mha_dimension, self.rolled_dim)])
        self.ff3 = torch.nn.Sequential(*[torch.nn.Linear(self.rolled_dim, self.rolled_dim*4), torch.nn.LeakyReLU(True), torch.nn.Linear(self.rolled_dim*4, self.rolled_dim)])
        self.dropout2 = torch.nn.Dropout(0.2)
        self.norm2 = torch.nn.LayerNorm(self.rolled_dim)
    def forward(self, query: "Trajectory", keyval: "Reference"):
        #Almost GPT like
        keyval = keyval.expand_as(query) #1LC -> BLC
        #print(query.shape, keyval.shape)
        Q = self.q(query) #B,L,1200
        K, V = self.kv(keyval).chunk(2, dim=-1) #B,L,1200
        Qh, Kh, Vh = list(map(lambda inp: einops.rearrange(inp, "b l (h d) -> b h l d", h=self.nheads), [Q, K, V])) #B,H,L,Dim
        mha_logits = torch.einsum("bhid, bhjd -> bhij", Qh, Kh)
        #print(mha_logits.shape, self.dim_per_head)
        mha_logits_ = mha_logits / (self.dim_per_head) #B,H,L,L
        mha_softmax = torch.nn.Softmax(dim=-1)(mha_logits_)
        mha = torch.einsum("bhij, bhjd -> bhid", mha_softmax, Vh)
        mha_ = einops.rearrange(mha, "b h i d -> b i (h d)") #B,L,1200
        output = self.ff(mha_) #B,L,D (original dim)
        #output = mha_ #B,L,D 
        output = self.dropout(output)
        output_ = self.norm(output + query) #B,L,D (residual)
        #output_ = self.norm(output + Q) #B,L,D (residual)
        output2 = self.ff2(output_) #B,L,D to B,L,3 ... 
        output3 = self.dropout2(output2) #B,L,3
        output3 = self.norm2(output_ + output3)
        output3 = output3 + keyval #Predict difference instead of coord...
        #output4 = self.ff3(output2) + keyval #B,L,3 Predict difference instead of coord...
        return output3, mha_softmax

class Encoder(torch.nn.Module):
    def __init__(self, hidden_dims=[1500, 750, 400, 200, 200], **kwargs):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.unrolled_dim = kwargs.get("unrolled_dim") #xyz coord dim of original protein trajectory

        linears = torch.nn.Sequential(*[ 
                                      torch.nn.Linear(self.unrolled_dim, self.hidden_dims[0]), torch.nn.BatchNorm1d(self.hidden_dims[0]), torch.nn.SiLU(), 
                                      torch.nn.Linear(self.hidden_dims[0], self.hidden_dims[1]), torch.nn.BatchNorm1d(self.hidden_dims[1]), torch.nn.SiLU(True),                 
                                      torch.nn.Linear(self.hidden_dims[1], self.hidden_dims[2]), torch.nn.BatchNorm1d(self.hidden_dims[2]), torch.nn.SiLU(True),
                                      torch.nn.Linear(self.hidden_dims[2], self.hidden_dims[3]), torch.nn.BatchNorm1d(self.hidden_dims[3]), torch.nn.SiLU(True),                
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
        return mu + logstd.exp() * torch.distributions.Normal(0., 0.3).rsample((shapes)).to(mu)
    def normalize(self, x):
        x_ = (x - x.mean(dim=0)) / x.std(dim=0)
        return x_

#enc = Encoder(unrolled_dim = trajectory.size(1)*trajectory.size(2))
#enc.to(device)
#z, _, _ = enc(trajectory)
#plt.scatter(*z.detach().cpu().numpy().T); plt.show()

class Decoder(torch.nn.Module):
    def __init__(self, hidden_dims=list(reversed([1500, 750, 400, 200, 100])), **kwargs):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.unrolled_dim = kwargs.get("unrolled_dim") #xyz coord dim of original protein trajectory
        self.rolled_dim = kwargs.get("rolled_dim") #xyz coord dim of original protein trajectory
        self.reference = kwargs.get("reference") #PDB of reference
        self.mha_dimension = kwargs.get("mha_dimension", 1200)
        self.nheads = kwargs.get("nheads", 6)
        self.layers = kwargs.get("layers", 6)

        linears = torch.nn.Sequential(*[ 
                                      torch.nn.Linear(self.hidden_dims[0], self.hidden_dims[1]), torch.nn.BatchNorm1d(self.hidden_dims[1]), torch.nn.SiLU(True),                 
                                      torch.nn.Linear(self.hidden_dims[1], self.hidden_dims[2]), torch.nn.BatchNorm1d(self.hidden_dims[2]), torch.nn.SiLU(True),
                                      torch.nn.Linear(self.hidden_dims[2], self.hidden_dims[3]), torch.nn.BatchNorm1d(self.hidden_dims[3]), torch.nn.SiLU(True),                
                                      torch.nn.Linear(self.hidden_dims[3], self.hidden_dims[4]), torch.nn.BatchNorm1d(self.hidden_dims[4]), torch.nn.SiLU(True),           
                                      torch.nn.Linear(self.hidden_dims[4], self.unrolled_dim), torch.nn.ELU(True)
                                    ]) #B,C,H,W
        self.add_module("linears_sequential", linears)
        self.mha_res = MultiheadAttention_Residual(rolled_dim=self.rolled_dim, mha_dimension=self.mha_dimension, nheads=self.nheads)
        feedforward = torch.nn.Sequential(*[torch.nn.Linear(self.rolled_dim, self.rolled_dim), torch.nn.LeakyReLU(True), torch.nn.Linear(self.rolled_dim, self.rolled_dim)])
        self.add_module("ff", feedforward)
        self.pos_emb = torch.nn.Embedding(self.reference.size(1), 3) #reference is (1,L,3)
    def forward(self, inputs: "BD"):
        sizes = self.reference.size() #1,L,3
        x = inputs #Latent dim
        x = self.linears_sequential(x)
        #pos_emb = self.pos_emb(torch.arange(self.reference.size(1))[None,:].to(x).long())
        x_q = x.view(x.size(0), sizes[1], sizes[2]) #+ pos_emb #B,L,3 
        #x_enc = self.reference + pos_emb #1,L,3
        #attns = {}
        #for i in range(self.layers):
        #    x_q, attn = self.mha_res(x_q, x_enc) #B,L,3
        #    attns.update(i=attn)
        #x = self.ff(x_q)
        return x_q #, attns
#dec = Decoder(reference=reference, rolled_dim=3, unrolled_dim=trajectory.size(1)*trajectory.size(2), mha_dimension=1200, nheads=6, layers=1)
#dec.to(device)
#x, attn = dec(z)
#print(x, attn)

class VAE(torch.nn.Module):
    #VAE github: https://github.com/AntixK/PyTorch-VAE/tree/master/models
    def __init__(self, **kwargs):
        super().__init__()
        self.hidden_dims_enc = kwargs.get("hidden_dims_enc", None)
        self.hidden_dims_dec = kwargs.get("hidden_dims_dec", None)
        self.unrolled_dim = kwargs.get("unrolled_dim") #xyz coord dim of original protein trajectory
        self.rolled_dim = kwargs.get("rolled_dim") #xyz coord dim of original protein trajectory
        self.reference = kwargs.get("reference") #PDB of reference
        self.mha_dimension = kwargs.get("mha_dimension", 1200)
        self.nheads = kwargs.get("nheads", 6)
        self.layers = kwargs.get("layers", 6)
        self.encoder = Encoder(hidden_dims=self.hidden_dims_enc, unrolled_dim=self.unrolled_dim)
        self.decoder = Decoder(hidden_dims=self.hidden_dims_dec, reference=self.reference, rolled_dim=self.rolled_dim, unrolled_dim=self.unrolled_dim, mha_dimension=self.mha_dimension, nheads=self.nheads, layers=self.layers)
        #self.apply(self._init_weights)
    def forward(self, inputs: "Trajectory"):
        x = inputs #Normalized input
        z, mu, logstd = self.encoder(x)
        x = self.decoder(z) #BL3, Dict: BHLL x Layers
        return z, mu, logstd, x
    def losses(self, inputs, z, mu, logstd, recon: "x"):
        rmsd = torch.sqrt(torch.mean((inputs - recon)**2, dim=(-1, -2))).mean() #rmsd
        mse = torch.nn.MSELoss(reduction="mean")(recon, inputs)
        kl = torch.mean(-0.5 * torch.sum(1 + logstd - mu ** 2 - logstd.exp(), dim = 1), dim = 0) #kl-div
        L = max(15, inputs.shape[-2])
        d0 = 1.24 * (L - 15)**(1/3) - 1.8
        # get distance
        dist = ((inputs - recon)**2).sum(dim=-1).sqrt()
        tm = (1 / (1 + (dist/d0)**2)).mean(dim=-1).mean() #TM-score
        inputs_mat = torch.cdist(inputs, inputs, p=2)
        recon_mat = torch.cdist(recon, recon, p=2)
        mat = torch.mean((inputs_mat - recon_mat)**2, dim=(-1, -2)).mean() #Pairwise distance loss
        #rot, cen = kabsch_torch(recon, inputs, cpu=False)
        #kbrmsd = torch.sqrt(torch.mean((cen - rot)**2, axis=(-1, -2))).mean() #Kansch rmsd
        return kl + rmsd + tm #+ kl - tm + mat  # + kbrmsd
    def _init_weights(self, m: torch.nn.Module):
        if isinstance(m, torch.nn.Linear):
            m.weight.data.normal_(1,0.5)
            m.bias.data.zero_()
        
#vae = VAE(hidden_dims_enc=[1500, 750, 300, 50, 4], hidden_dims_dec=[2, 50, 300, 750, 1500], reference=reference, rolled_dim=3, unrolled_dim=trajectory.size(1)*trajectory.size(2), mha_dimension=1200, nheads=6, layers=3)
#vae.to(device)
#z, mu, logstd, recon, attn = vae(trajectory)
#print(recon, vae.losses(trajectory, z, mu, logstd, recon, attn))

###############################
##########TRAINING#############
###############################
def train(reference, trajectory, trajectoryv, CurrentModel, saved_from=None, save_to=None, epochs=50, continued=False):
    os.chdir("/Scr/hyunpark/Monster/")
    #MULTIHEAD is VERY important!!!
    torch.manual_seed(42)
    model = CurrentModel(hidden_dims_enc=[1500, 1000, 750, 300, 4], hidden_dims_dec=[2, 300, 750, 1000, 1500], reference=reference, rolled_dim=3, unrolled_dim=trajectory.size(1)*trajectory.size(2), mha_dimension=1200, nheads=6, layers=8)
    if continued: 
        ckpt = torch.load(saved_from, map_location=device)
        model.load_state_dict(ckpt)
    dataset = torch.utils.data.TensorDataset(trajectory)   #Only easy trajs (closed states)
    datasetv = torch.utils.data.TensorDataset(trajectoryv) #
    #dataset_total = torch.utils.data.ConcatDataset([dataset, datasetv])
    lend = len(dataset)
    lendv = len(datasetv)
    #del dataset
    #del datasetv
    #gc.collect()
    #dataset, datasetv = torch.utils.data.random_split(dataset_total, [lend + lendv - 40, 40])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
    dataloaderv = torch.utils.data.DataLoader(datasetv, batch_size=16, shuffle=True)    
    del dataset
    del datasetv
    #del dataset_total
    gc.collect()

    checks = [(name, p.requires_grad) for name, p in model.named_parameters()]
    print(fmt.blue(f"Checking for frozen and trainable parameters: {checks}..."))
    optim = torch.optim.AdamW(model.parameters()) #Prev frozen plus current

    model.to(device)
    best_valloss = float("Inf")
    for e in range(epochs):
        model.train()
        valloss = 0.
        if e >= 100 and e%5 == 0:
            optim.param_groups[0]['lr'] = optim.param_groups[0]['lr'] * 0.95
        for idx, traj in enumerate(dataloader):
            traj = traj[0].to(device)
            model.train()
            optim.zero_grad()
            z, mu, logstd, recon_ = model(traj)
            #print(z, recon)
            loss = model.losses(traj, z, mu, logstd, recon_)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.8)
            optim.step()
        for traj in dataloaderv:
            traj = traj[0].to(device)
            model.eval()
            z, mu, logstd, recon = model(traj)
            loss = model.losses(traj, z, mu, logstd, recon)
            valloss += loss.item()
        #best_recon = recon
        current_valloss = valloss/len(dataloaderv)
        if current_valloss < best_valloss:
            best_valloss = current_valloss
            torch.save(model.state_dict(), save_to)
            fake_closed = mda.Universe(CRD)
            fake_closed_ca = fake_closed.atoms.select_atoms(atom_selection)
            if e >= 1: 
                recon = unnormalize(recon, mean=mean, std=std)
                recon_ = unnormalize(recon_, mean=mean, std=std)
                fake_closed_ca.atoms.positions = recon[0].detach().cpu().numpy()
                fake_closed_ca.write("fake0.pdb")
                fake_closed_ca.atoms.positions = recon[1].detach().cpu().numpy()
                fake_closed_ca.write("fake1.pdb")
                fake_closed_ca.atoms.positions = recon_[0].detach().cpu().numpy()
                fake_closed_ca.write("fake_0.pdb")
                fake_closed_ca.atoms.positions = recon_[1].detach().cpu().numpy()
                fake_closed_ca.write("fake_1.pdb")
        print(fmt.yellow(f"At epoch {e}, mean validation loss is {current_valloss}..."))
        #torch.save(monster.state_dict(), save_to)
    return model, recon
#_, recon = train(reference, trajectory, trajectoryv, CurrentModel=VAE, saved_from="New_VAE.pt", save_to="New_VAE.pt", epochs=300, continued=False)


###############################
##########TESTING##############
###############################
if __name__ == "__main__":
    model = VAE(hidden_dims_enc=[1500, 1000, 750, 300, 4], hidden_dims_dec=[2, 300, 750, 1000, 1500], reference=reference, rolled_dim=3, unrolled_dim=trajectory.size(1)*trajectory.size(2), mha_dimension=1200, nheads=6, layers=8)
    ckpt = torch.load("New_VAE.pt", map_location=device)
    model.load_state_dict(ckpt)
    model.to(device)
    model.eval()

    def lerp(inputs: "1D tensor of features", outputs: "1D tensor of features", interps: "1D tensor of weights"):
        outs = inputs + (outputs - inputs) * interps.view(-1,1).to(inputs)
        return outs

    traj = torch.cat((trajectory, trajectoryv), dim=0)
    z, mu, logstd, recon = model(traj)
    #plt.scatter(*mu[:102].T.detach().cpu().numpy(), c=np.arange(102)); plt.scatter(*mu[102:].T.detach().cpu().numpy(), c=np.arange(98)); plt.colorbar(); plt.scatter([0.2], [-0.]); plt.show()

    interolations = 100
    lerps = lerp(mu[0], mu[-1], torch.linspace(0,1,interolations)[1:-1])
    lerps2 = lerp(torch.tensor([-0.3,-0.2]).to(device), torch.tensor([0.3,0.]).to(device), torch.linspace(0,1,interolations)[1:-1])
    x = model.decoder(lerps.to(mu))
    recon_ = unnormalize(x, mean=mean, std=std)
    x = model.decoder(lerps2.to(mu))
    recon2_ = unnormalize(x, mean=mean, std=std)
    x = model.decoder(z.to(mu))
    RECON = unnormalize(x, mean=mean, std=std)
    fake_closed = mda.Universe(CRD)
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
    


