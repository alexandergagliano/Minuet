import torch
import torch.nn as nn
import h5py
import glob
import pyro
import pyro.infer as infer
import pyro.distributions as dist
from pyro.infer.mcmc import MCMC, NUTS
import numpy as np
import matplotlib.pyplot as plt
from daep.LitWrapper import LitMaskeddaep
import tarp
import zuko
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import pandas as pd

def plot_config():
    sns.set_context("poster")
    sns.set(font_scale=2)
    sns.set_style("white")
    sns.set_style("ticks", {"xtick.major.size": 15, "ytick.major.size": 20})
    sns.set_style("ticks", {"xtick.minor.size": 8, "ytick.minor.size": 8})
    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palantino"],
    "xtick.minor.visible": True,     # Make minor ticks visible on the x-axis
    "ytick.minor.visible": True,     # Make minor ticks visible on the y-axis
    "xtick.direction": "in",         # Minor ticks direction for x-axis
    "ytick.direction": "in",         # Minor ticks direction for y-axis
    "xtick.top": True,               # Show ticks on the top x-axis
    "ytick.right": True,             # Show ticks on the right y-axis
    "xtick.major.size": 15,          # Major tick size for x-axis
    "ytick.major.size": 20,          # Major tick size for y-axis
    "xtick.minor.size": 8,           # Minor tick size for x-axis
    "ytick.minor.size": 8,           # Minor tick size for y-axis
    })

plot_config()

latent_dim = 5
theta_dim = 3

print("Loading data...")

val_set = h5py.File("/Users/alexgagliano/Documents/Research/GalaxyAutoencoder/data/test_subset/maskeddaep5_results_sample_1753214750_maskdaep5.h5", 'r')

# phase 1: predicting latent params from redshift, stellar mass, and SFR
latents = val_set['latent'][:]
theta = val_set['y_true'][:, 0, 0:3]

mask = (theta[:,0] > 0) & (theta[:,0] < 5)  # Redshift between 0 and 5
mask &= (theta[:,1] > 5) & (theta[:,1] < 13)  # log(M*/Msun) between 5 and 13
mask &= (theta[:,2] > -3) & (theta[:,2] < 3)  # SFR between -3 and 3 (if log scale)

latents = latents[mask]
theta = theta[mask]

#normalize
theta_mean = theta.mean(axis=0)
theta_std = theta.std(axis=0)
theta_norm = (theta - theta_mean) / theta_std

latents_train, latents_test, theta_train, theta_test = train_test_split(latents, theta_norm, test_size=0.2, random_state=42)

latents_train = torch.tensor(latents_train, dtype=torch.float32).reshape(-1, latent_dim)
latents_test = torch.tensor(latents_test, dtype=torch.float32).reshape(-1, latent_dim)

theta_train = torch.tensor(theta_train, dtype=torch.float32)
theta_test = torch.tensor(theta_test, dtype=torch.float32)

print("z_data shape:", latents_train.shape)
print("theta_data:", theta_train.shape)

# Fix flow initialization - context dim should be latent_dim, target dim should be theta_dim
flow = zuko.flows.NSF(theta_dim, latent_dim, transforms=10, hidden_features=[512] * 6)

saved = glob.glob("./daep_conditional_flow_3param_5latent.pt")
if len(saved) > 0:
    state_dict = torch.load(saved[-1], map_location='cpu')
    flow.load_state_dict(state_dict)
    print("Saved flow loaded.")


# Fix test_loader scope - define it outside the training block
test_dataset = torch.utils.data.TensorDataset(latents_test, theta_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

# evaluate performance of flow on test set
import torch
import numpy as np

# Collect predictions and ground truths
all_theta_sampled = []
all_theta_true = []

with torch.no_grad():
    for i, (z_batch, theta_batch) in enumerate(test_loader):
        theta_sampled = flow(z_batch).sample((100,))  
        all_theta_sampled.append(theta_sampled.cpu())  # detach from GPU
        all_theta_true.append(theta_batch.cpu())
        print(f"Predictions generated for Batch {i}...")

# Stack everything
# theta_sampled: [64, N_total, dim]
theta_sampled = torch.cat(all_theta_sampled, dim=1).numpy()
theta_true = torch.cat(all_theta_true, dim=0).numpy()

# Call TARP once on full set
coverage = tarp.get_tarp_coverage(
    theta_sampled,        # shape [64, N, d]
    theta_true,           # shape [N, d]
    references='random',
    metric='euclidean',
    num_alpha_bins=100,
    num_bootstrap=100,
    norm=False,
    bootstrap=False,
    seed=42
)


alpha, emp_coverage = coverage

df = pd.DataFrame({'alpha':alpha, 'emp_coverage':emp_coverage})
df.to_csv("./empiricalcoverage.csv", index=False)

plt.figure(figsize=(6, 6))
plt.plot(alpha, emp_coverage, label='Empirical Coverage')
plt.plot(alpha, alpha, '--', color='gray', label='Ideal (y=x)')
plt.xlabel(r"Nominal Confidence Level ($\alpha$)")
plt.ylabel("Empirical Coverage")
plt.title("TARP Coverage Plot")
plt.grid(True)
plt.tight_layout()
plt.savefig("./coverage_paramflow.png", dpi=300, bbox_inches='tight')
