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
import zuko
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns

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

#val_set = h5py.File("/n/holylabs/LABS/iaifi_lab/Users/agagliano/galaxyAutoencoder/test_results/maskeddaep_results_1751500541_maskdaep.h5", 'r')
#val_set = h5py.File("/Users/alexgagliano/Documents/Research/GalaxyAutoencoder/data/test_subset/maskeddaep_results_1751500541_maskdaep.h5", 'r')
val_set = h5py.File("/Users/alexgagliano/Documents/Research/GalaxyAutoencoder/data/test_subset/maskeddaep5_results_sample_1752848125_maskdaep5.h5", 'r')
#ckpt_path = "/Users/alexgagliano/Documents/Research/GalaxyAutoencoder/models/models/fullTrain_maskdaep/last.ckpt"
#ckpt_path = "/n/holylabs/LABS/iaifi_lab/Users/agagliano/galaxyAutoencoder/models/fullTrain_maskdaep/last.ckpt"

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

#check for saved flow -- commented out for now
#saved = glob.glob("./*flow.pt")
saved = []
if len(saved) > 0:
    state_dict = torch.load(saved[-1], map_location='cpu')
    flow.load_state_dict(state_dict)
    print("Saved flow loaded.")
else:
    #train 
    optimizer = torch.optim.Adam(flow.parameters(), lr=1e-4)
    
    num_epochs = 50
    batch_size = 128
    
    train_dataset = torch.utils.data.TensorDataset(latents_train, theta_train)
    test_dataset = torch.utils.data.TensorDataset(latents_test, theta_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    print("Data loaded, training flow!")
    
    for epoch in range(num_epochs):
        for z_batch, theta_batch in train_loader:
            log_prob = flow(z_batch).log_prob(theta_batch)
            
            loss = -log_prob.mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        #if epoch % 50 == 0:  # Print less frequently
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    # save model 
    torch.save(flow.state_dict(), "daep_conditional_flow_3param_5latent.pt")

    print("Trained flow saved!")

# Fix test_loader scope - define it outside the training block
test_dataset = torch.utils.data.TensorDataset(latents_test, theta_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

# evaluate performance of flow on test set
all_theta_true = []
all_theta_pred = []

print("Evaluating flow performance...")
with torch.no_grad():
    for z_batch, theta_batch in test_loader:
        theta_sampled = flow(z_batch).sample((64,))
        theta_pred = theta_sampled.mean(dim=0)  # Average over samples, keep batch dimension 
        
        all_theta_true.append(theta_batch)
        all_theta_pred.append(theta_pred)

theta_true = torch.cat(all_theta_true, dim=0)
theta_pred = torch.cat(all_theta_pred, dim=0)

print("shape theta_true:", theta_true.shape)
print("shape theta_pred:", theta_pred.shape)

for i in range(theta_dim):
    #re-normalize
    theta_pred[:, i] *= theta_std[i]
    theta_pred[:, i] += theta_mean[i]
 
    theta_true[:, i] *= theta_std[i]
    theta_true[:, i] += theta_mean[i]

    # Calculate residuals and define outliers
    residuals = theta_pred[:, i] - theta_true[:, i]
    residuals = residuals.numpy()

    threshold = 3 * np.std(residuals)
    is_outlier = np.abs(residuals) > threshold

    fig = plt.figure(figsize=(10, 7))
    # Plot central density (excluding outliers)
    hb = plt.hexbin(
        theta_true[~is_outlier, i], theta_pred[~is_outlier, i],
        gridsize=300, cmap='inferno', bins='log', mincnt=1, alpha=0.8
    )
    # Overlay outliers as hollow circles
    plt.plot(
        theta_true[is_outlier, i], theta_pred[is_outlier, i],
        'o', markeredgecolor='k', markerfacecolor='none', markersize=6, alpha=0.7, label='Outliers'
    )
    # 1:1 reference line
    plt.plot(
        [theta_true[:, i].min(), theta_true[:, i].max()],
        [theta_true[:, i].min(), theta_true[:, i].max()],
        c='tab:blue', ls='--'
    )

    mse = mean_squared_error(theta_true[:, i], theta_pred[:, i])
    r2 = r2_score(theta_true[:, i], theta_pred[:, i])

    plt.xlabel(r'Zou+22 $\theta$')
    plt.ylabel(r'Predicted $\theta$')
    plt.title(f'MSE={mse:.4f}, R2={r2:.4f}')
    if i==2:
        plt.xlim((-3, 2))
        plt.ylim((-3, 2))
    cb = plt.colorbar(hb)
    cb.set_label('log10(N)')
    #plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(f'theta_pred_{i}_5latent.png')
    plt.close()

    print(f"MSE for parameter {i}: {mse:.4f}")
    print(f"R2 for parameter {i}: {r2:.4f}")
