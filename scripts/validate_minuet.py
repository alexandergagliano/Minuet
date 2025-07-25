import h5py
from itertools import combinations
import gc
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import torch
import sys
import pandas as pd
from tqdm import tqdm
from itertools import combinations
sys.path.append("/Users/alexgagliano/Documents/Research/GalaxyAutoencoder/code/ssl-legacysurvey/")
from ssl_legacysurvey.data_loaders import decals_augmentations
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.colors as mcolors
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from scipy import stats
from umap import UMAP
import hdbscan
import cv2
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
import gc
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from scipy.stats import binned_statistic_2d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import gc
from sklearn.manifold import SpectralEmbedding
from scipy.optimize import linear_sum_assignment
from PIL import Image
from sklearn.neighbors import KDTree

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

# Add model path to system path
sys.path.append("/Users/alexgagliano/Documents/Research/GalaxyAutoencoder/models/")
#from cvae_model import *

# Set matplotlib to non-interactive backend
plt.switch_backend('agg')
NDIM = 10

def load_models(base_dir):
    """Get dictionary of model paths and their corresponding checkpoints"""
    test_subset_dir = os.path.join(base_dir, "data/test_subset/")
    models_dir = os.path.join(base_dir, "models/models")

    models = {'masked_daep_5': {'test_results': os.path.join(test_subset_dir, "maskeddaep5_results_sample_1753214750_maskdaep5.h5")}}
    return models

def plot_reconstruction_comparison(model, save_dir, title, ngals=20, shown_idxs=None):
    """Plot reconstruction comparison for a single model with improved visualization

    Args:
        model: Dictionary containing model data
        save_dir: Directory to save plots
        title: Title for the plot
        ngals: Number of galaxies to show
        shown_idxs: Optional list of specific indices to show
    """

    Nbright = len(model['bright']['images'])
    if shown_idxs is None:
        shown_idxs = [np.random.randint(0, Nbright) for _ in range(ngals)]

    fig, axis = plt.subplots(4, ngals, figsize=(20,4), sharey=True, sharex=True)

    # Define normalization for residuals
    def normalize_residual(diff):
        # Normalize to [0,1] range with sqrt scaling to enhance visibility
        return np.clip(np.sqrt(np.abs(diff)) / np.sqrt(np.abs(diff).max()), 0, 1)

    for i, idx in enumerate(shown_idxs):
        #try:
        if True:
            # Load and process images
            real_img = np.array(model['bright']['images'][idx], dtype=np.float32)
            real_img = np.moveaxis(real_img, 0, -1)

            print(model['bright']['recon'].shape)
            gen_img = np.array(model['bright']['recon'][idx, 0, :, :, :], dtype=np.float32)
            gen_img = np.moveaxis(gen_img, 0, -1)  # Move channels to last dimension

            mask_img = np.array(model['bright']['seg_mask'][idx], dtype=np.float32)
            #mask_img = np.moveaxis(mask_img, 0, -1)

            # Calculate residuals in a more informative way
            diff_img = np.abs(gen_img - real_img)

            # Create normalized residual image that highlights structural differences
            residual = np.zeros_like(diff_img)
            for c in range(3):  # Process each channel
                residual[:,:,c] = normalize_residual(diff_img[:,:,c])

            # Plot images with proper normalization
            axis[0, i].imshow(real_img)
            axis[1, i].imshow(mask_img, cmap='Greys_r')  # Only use first channel for mask
            axis[2, i].imshow(gen_img)
            axis[3, i].imshow(residual, cmap='magma')

            for j in range(4):
                axis[j, i].set_xticks([])
                axis[j, i].set_yticks([])

            # Clean up
            del real_img, gen_img, mask_img, diff_img, residual
            gc.collect()

    axis[0, 0].set_ylabel("Original")
    axis[1, 0].set_ylabel("Segmentation")
    axis[2, 0].set_ylabel("Reconstruction")
    axis[3, 0].set_ylabel("Residual")

    plt.suptitle(title)#, fontsize=20)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.savefig(os.path.join(save_dir, f"{title.lower().replace(' ', '_')}.png"), dpi=300, bbox_inches='tight')
    plt.close()
    gc.collect()

    return shown_idxs

def plot_latent_physical_correlations(model, save_dir, param_idx, param_name, nbins=500):
    """Plot correlations between latent space and physical parameters"""
    # Use only bright galaxies for consistency
    params = np.squeeze(model['y_true'][:])
    #mu = model['latent'][:].reshape(NDIM, -1).T
    mu = model['latent'][:]
    mu = mu.reshape(len(mu), -1)

    x = params[:, param_idx]
    y = mu[:, param_idx]

    fig, ax = plt.subplots(2, 1, figsize=(7, 10), sharex=True, height_ratios=(2, 1))

    # Plot correlation
    H, xedges, yedges = np.histogram2d(x, y, bins=nbins)
    H = np.rot90(H)
    H = np.flipud(H)
    Hmasked = np.ma.masked_where(H==0, H)

    mesh = ax[0].pcolormesh(xedges, yedges, Hmasked, cmap='plasma')
    ax[0].plot([-2, 0, 2], [-2, 0, 2], ls='--', lw=2, c='k', alpha=0.3)
    ax[0].set_ylabel(f"Latent Feature {param_idx+1}")
    ax[0].set_xlim((-2, 2))
    ax[0].set_ylim((-2, 2))

    # Plot residuals
    y_diff = y - x
    H, xedges, yedges = np.histogram2d(x, y_diff, bins=nbins)
    H = np.rot90(H)
    H = np.flipud(H)
    Hmasked = np.ma.masked_where(H==0, H)

    mesh = ax[1].pcolormesh(xedges, yedges, Hmasked, cmap='plasma')
    ax[1].set_xlabel(f"True {param_name}")
    ax[1].set_ylabel("Latent - True")
    ax[1].set_ylim(-2, 2)
    ax[1].axhline(y=0, lw=2, c='k')

    plt.savefig(os.path.join(save_dir, f"latent_correlation_{param_name.lower()}.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_latent_space_with_properties(model, save_dir, bright_only=False):
    """Create scatter plots of latent space colored by physical properties"""
    if bright_only:
        # Use only bright galaxies
        mu = model['bright']['latent'][:]
        mu = mu.reshape(len(mu), -1)

        z1 = mu[:, 0]
        z2 = mu[:, 1]
        params = np.squeeze(model['bright']['y_true'][:])
    else:
        # Use all galaxies
        #z1 = model['latent'][:].reshape(NDIM, -1).T[:, 0]
        #z2 = model['latent'][:].reshape(NDIM, -1).T[:, 1]

        mu = model['latent'][:]
        mu = mu.reshape(len(mu), -1)

        z1 = mu[:, 0]
        z2 = mu[:, 1]

        params = np.squeeze(model['y_true'][:])

    fig, axes = plt.subplots(1, 3, figsize=(12, 15))

    # Plot redshift distribution
    redshift = params[:, 0]
    sc = axes[0].scatter(z1, z2, c=redshift, cmap='viridis', s=1, alpha=0.5)
    #axes[0].set_title('Latent Space Colored by Redshift')
    plt.colorbar(sc, ax=axes[0])

    # Plot stellar mass distribution
    stellar_mass = params[:, 1]
    sc = axes[1].scatter(z1, z2, c=stellar_mass, cmap='magma', s=1, alpha=0.5)
    #axes[1].set_title('Latent Space Colored by Stellar Mass')
    plt.colorbar(sc, ax=axes[1])

    sfr = params[:, 2]
    sc = axes[1].scatter(z1, z2, c=sfr, cmap='inferno', s=1, alpha=0.5)
    #axes[1].set_title('Latent Space Colored by SFR')
    plt.colorbar(sc, ax=axes[2])

    for ax in axes:
        ax.set_xlabel('Latent Dimension 1')
        ax.set_ylabel('Latent Dimension 2')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'latent_space_properties{"_bright" if bright_only else "_all"}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_latent_correlations(model, save_dir):
    """Plot pairwise correlations between latent dimensions"""
    #mu = model['latent'][:].reshape(NDIM, -1).T
    mu = model['latent'][:]
    mu = mu.reshape(len(mu), -1)

    df = pd.DataFrame(mu, columns=[f'z{i}' for i in range(mu.shape[1])])
    g = sns.pairplot(df, corner=True, plot_kws={'s': 5, 'alpha': 0.3})
    g.fig.suptitle("Latent Space Correlations")#, y=1.02)
    plt.savefig(os.path.join(save_dir, 'latent_correlations.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_galaxies_in_latent_space(model, save_dir, n_samples=1000, thumbnail_size=0.6):
    """Plot galaxy thumbnails at their actual positions in latent space

    Args:
        model: Dictionary containing model data
        save_dir: Directory to save plots
        n_samples: Number of thumbnail samples to show
        thumbnail_size: Size of thumbnails relative to figure (smaller values = smaller thumbnails)
    """
    #mu = model['latent'][:].reshape(NDIM, -1).T
    mu = model['latent'][:]
    mu = mu.reshape(len(mu), -1)

    n_dims = mu.shape[1]

    # Get all pairs of dimensions
    dim_pairs = list(combinations(range(n_dims), 2))

    # Define dimension names for better labeling
    dim_names = ["Dim1", "Dim2", "Dim3", "Dim4", "Dim5", "Dim6", "Dim7", "Dim8", "Dim9", "Dim10"]

    # For each pair of dimensions
    for dim1, dim2 in tqdm(dim_pairs, desc="Plotting latent dimension pairs"):
        # Extract data for these dimensions
        z1 = mu[:, dim1]
        z2 = mu[:, dim2]

        # Create figure
        plt.figure(figsize=(24, 24))

        # Plot background points
        plt.scatter(z1, z2, c='gray', s=2, alpha=0.1)

        # Calculate axis limits and add padding
        x_min, x_max = z1.min(), z1.max()
        y_min, y_max = z2.min(), z2.max()

        x_padding = 0.05 * (x_max - x_min)
        y_padding = 0.05 * (y_max - y_min)

        plt.xlim(x_min - x_padding, x_max + x_padding)
        plt.ylim(y_min - y_padding, y_max + y_padding)

        # Choose points to display - use grid sampling for even coverage
        n_gridx = int(np.sqrt(n_samples))
        n_gridy = int(np.sqrt(n_samples))

        x_bins = np.linspace(x_min, x_max, n_gridx + 1)
        y_bins = np.linspace(y_min, y_max, n_gridy + 1)

        thumbnails_placed = 0
        for i in range(n_gridx):
            for j in range(n_gridy):
                # Find points in this grid cell
                in_cell = (z1 >= x_bins[i]) & (z1 < x_bins[i+1]) & \
                          (z2 >= y_bins[j]) & (z2 < y_bins[j+1])

                if np.sum(in_cell) > 0:
                    # Pick a random point from this cell
                    cell_indices = np.where(in_cell)[0]
                    idx = np.random.choice(cell_indices)

                    try:
                        # Get galaxy image
                        img = np.array(model['images'][idx], dtype=np.float32)
                        img = np.moveaxis(img, 0, -1)
                        #img = np.array(model['bright']['recon'][idx, 0, :, :, :], dtype=np.float32)
                        #img = np.moveaxis(img, 0, -1)  # Move channels to last dimension

                        # Create offsetbox
                        imagebox = OffsetImage(img, zoom=thumbnail_size)

                        # Position the thumbnail
                        ab = AnnotationBbox(imagebox, (z1[idx], z2[idx]),
                                          frameon=True, pad=0.0,
                                          bboxprops=dict(edgecolor='black'))

                        # Add to plot
                        plt.gca().add_artist(ab)
                        thumbnails_placed += 1
                    except Exception as e:
                        print(f"Error placing thumbnail at ({z1[idx]}, {z2[idx]}): {e}")

        print(f"Placed {thumbnails_placed} thumbnails for dimensions {dim1+1} vs {dim2+1}")

        # Add labels and title
        plt.xlabel(f"{dim_names[dim1]}")
        plt.ylabel(f"{dim_names[dim2]}")
        plt.title(f"{dim_names[dim1]} vs {dim_names[dim2]} (reconstructed)")#, fontsize=20)

        # Save and close
        plt.savefig(os.path.join(save_dir, f"latent_thumbnails_{dim1+1}_vs_{dim2+1}.png"),
                   dpi=300, bbox_inches='tight')
        plt.close()
        gc.collect()

def plot_galaxies_in_umap_space(model, save_dir, n_samples=2000, thumbnail_size=0.4):
    """Plot galaxy thumbnails positioned in UMAP 2D embedding space

    Args:
        model: Dictionary containing model data
        save_dir: Directory to save plots
        n_samples: Number of thumbnail samples to show
        thumbnail_size: Size of thumbnails relative to figure
    """
    from umap import UMAP

    # Get latent space and images
    #mu = model['latent'][:].reshape(NDIM, -1).T
    mu = model['latent'][:]
    mu = mu.reshape(len(mu), -1)

    images = model['images'][:]
    recons = model['recon'][:]
    params = np.squeeze(model['y_true'][:])
    all_idxs = np.arange(len(mu))

    if len(mu) > 50000:
        print("Subsetting to 50k galaxies...")
        chosen = np.random.choice(all_idxs, size=50000)
        # Sort indices for h5py compatibility
        chosen = np.sort(chosen)
        mu = mu[chosen, :]
        recons = recons[chosen, :, :, :]
        images = images[chosen, :, :, :]
        params = params[chosen, :]

    print(f"Computing UMAP embedding for {len(mu)} galaxies...")

    # Compute UMAP embedding
    umap_reducer = UMAP(n_neighbors=1000, min_dist=1, n_components=2, random_state=42)
    embedding = umap_reducer.fit_transform(mu)

    # Create plots for both original images and reconstructions
    for img_type, img_data in [('original', images), ('reconstruction', recons)]:
        print(f"Creating UMAP thumbnail plot for {img_type} images...")

        # Create figure
        plt.figure(figsize=(20, 20))

        # Plot all points as background
        plt.scatter(embedding[:, 0], embedding[:, 1], c='lightgray', s=1, alpha=0.3)

        # Calculate axis limits and add padding
        x_min, x_max = embedding[:, 0].min(), embedding[:, 0].max()
        y_min, y_max = embedding[:, 1].min(), embedding[:, 1].max()

        x_padding = 0.05 * (x_max - x_min)
        y_padding = 0.05 * (y_max - y_min)

        plt.xlim(x_min - x_padding, x_max + x_padding)
        plt.ylim(y_min - y_padding, y_max + y_padding)

        # Sample points for thumbnails using grid-based approach
        n_gridx = int(np.sqrt(n_samples))
        n_gridy = int(np.sqrt(n_samples))

        x_bins = np.linspace(x_min, x_max, n_gridx + 1)
        y_bins = np.linspace(y_min, y_max, n_gridy + 1)

        thumbnails_placed = 0
        for i in range(n_gridx):
            for j in range(n_gridy):
                # Find points in this grid cell
                in_cell = (embedding[:, 0] >= x_bins[i]) & (embedding[:, 0] < x_bins[i+1]) & \
                          (embedding[:, 1] >= y_bins[j]) & (embedding[:, 1] < y_bins[j+1])

                if np.sum(in_cell) > 0:
                    # Pick a random point from this cell
                    cell_indices = np.where(in_cell)[0]
                    idx = np.random.choice(cell_indices)

                    try:
                        # Get galaxy image
                        img = np.array(img_data[idx], dtype=np.float32)
                        # Handle the case where img has shape (channel, px_x, px_y) - just move channels to last dimension
                        img = np.moveaxis(img, 0, -1)  # Move channels to last dimension

                        # Create offsetbox
                        imagebox = OffsetImage(img, zoom=thumbnail_size)

                        # Position the thumbnail at UMAP coordinates
                        ab = AnnotationBbox(imagebox, (embedding[idx, 0], embedding[idx, 1]),
                                          frameon=True, pad=0.0,
                                          bboxprops=dict(edgecolor='black', linewidth=0.5))

                        # Add to plot
                        plt.gca().add_artist(ab)
                        thumbnails_placed += 1
                    except Exception as e:
                        print(f"Error placing thumbnail at UMAP coordinates: {e}")

        print(f"Placed {thumbnails_placed} thumbnails in UMAP space")

        # Add labels and title
        plt.xlabel('UMAP Dim 1')
        plt.ylabel('UMAP Dim 2')
        plt.title(f'Galaxy {img_type.title()}s\n')

        # Remove ticks for cleaner look
        plt.xticks([])
        plt.yticks([])

        # Save plot
        plt.savefig(os.path.join(save_dir, f'umap_galaxy_thumbnails_{img_type}.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
        gc.collect()

    # Also create a version colored by physical properties
    print("Creating UMAP plots colored by physical properties...")

    properties = {
        'redshift': {
            'data': params[:, 0],
            'cmap': 'viridis',
            'label': 'Redshift'
        },
        'mass': {
            'data': params[:, 1],
            'cmap': 'magma',
            'label': 'log($M*/M_{\odot}$)'
        },
        'sfr': {
            'data': params[:, 2],
            'cmap': 'inferno',
            'label': 'log(SFR/Gyr)'
        }
    }

    fig, axes = plt.subplots(1, 3, figsize=(20, 8))

    for i, (prop_name, prop) in enumerate(properties.items()):
        scatter = axes[i].scatter(embedding[:, 0], embedding[:, 1],
                               c=prop['data'],
                               cmap=prop['cmap'],
                               s=2, alpha=0.6)
        plt.colorbar(scatter, ax=axes[i], label=prop['label'])
        axes[i].set_xlabel('UMAP Dimension 1')
        axes[i].set_ylabel('UMAP Dimension 2')
        axes[i].set_title(f'UMAP Embedding Colored by {prop["label"]}')
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'umap_physical_properties.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_latent_dimension_traversal(model, cvae, save_dir):
    """Plot traversal through latent dimensions using the CVAE model, traversing by standard deviations"""

    #mu = model['bright']['latent'][:].reshape(NDIM, -1).T
    mu = model['bright']['latent'][:]
    mu = mu.reshape(len(mu), -1)

    # Get latent statistics for each dimension
    z_mean = torch.tensor(mu.mean(axis=0))
    z_std = torch.tensor(mu.std(axis=0))

    # Set up traversal parameters
    n_steps = 10  # number of steps
    n_sigma = 5  # number of standard deviations to traverse

    # Create figure
    d = z_mean.shape[0]  # number of latent dimensions
    fig, axes = plt.subplots(d, n_steps, figsize=(2*n_steps, 2*d))
    if d == 1:
        axes = axes.reshape(1, -1)

    cvae.eval()
    with torch.no_grad():
        for i in range(d):  # for each latent dimension
            # Create range from -n_sigma to +n_sigma
            sigma_range = np.linspace(-n_sigma, n_sigma, n_steps)

            for j, n_sig in enumerate(sigma_range):
                # Create latent vector starting from mean of all dimensions
                z = z_mean.clone()
                # Set the current dimension to mean + n_sigma * std
                z[i] = z_mean[i] + n_sig * z_std[i]

                # Generate image
                img = cvae.decode(z.unsqueeze(0).to(cvae.device))
                img = img.squeeze(0).cpu().permute(1, 2, 0).numpy()

                # Plot
                ax = axes[i, j]
                ax.imshow(img)
                ax.axis('off')

                # Add value labels
                if i == 0:
                    ax.set_title(f'{n_sig:+.1f}σ')
                if j == 0:
                    ax.set_ylabel(f'Latent {i+1}')

    plt.suptitle(r'Latent Space ($\sigma$) Traversal')#, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'latent_traversal.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_umap(model, save_dir, max_background_points=100000):
    """Create UMAP projection colored by physical properties"""
    from umap import UMAP

    #mu = model['latent'][:].reshape(NDIM, -1).T
    mu = model['latent'][:]
    mu = mu.reshape(len(mu), -1)

    params = np.squeeze(model['y_true'][:])

    print(f"Computing UMAP embedding for physical property visualization...")

    # Subsample points if needed for UMAP computation
    if len(mu) > max_background_points:
        idx = np.random.choice(len(mu), max_background_points, replace=False)
        # Sort indices for h5py compatibility
        idx = np.sort(idx)
        mu_sample = mu[idx]
        params_sample = params[idx]
    else:
        mu_sample = mu
        params_sample = params

    # Compute UMAP embedding
    umap_reducer = UMAP(n_neighbors=1000, min_dist=1, n_components=2, random_state=42)
    embedding = umap_reducer.fit_transform(mu_sample)

    # Properties to color by
    properties = {
        'redshift': {
            'data': params_sample[:, 0],
            'cmap': 'viridis',
            'label': 'Redshift',
            'vmin': np.percentile(params_sample[:, 0], 5),   # 5th percentile
            'vmax': np.percentile(params_sample[:, 0], 95)   # 95th percentile
        },
        'mass': {
            'data': params_sample[:, 1],
            'cmap': 'magma',
            'label': 'log($M*/M_{\odot}$)',
            'vmin': np.percentile(params_sample[:, 1], 5),   # 5th percentile
            'vmax': np.percentile(params_sample[:, 1], 95)   # 95th percentile
        },
        'sfr': {
            'data': params_sample[:, 2],
            'cmap': 'inferno',
            'label': 'log(SFR/Gyr)',
            'vmin': -2, #np.percentile(params_sample[:, 2], 5),   # 5th percentile
            'vmax': np.percentile(params_sample[:, 2], 95)   # 95th percentile
        },
    }

    # Create main UMAP plots colored by physical properties
    fig, axes = plt.subplots(3, 1, figsize=(12, 16))

    for i, (prop_name, prop) in enumerate(properties.items()):
        # Scatter plot colored by property
        scatter = axes[i].scatter(embedding[:, 0], embedding[:, 1],
                                   c=prop['data'],
                                   cmap=prop['cmap'],
                                   s=3, alpha=0.7,
                                   vmin=prop['vmin'],
                                   vmax=prop['vmax'])
        plt.colorbar(scatter, ax=axes[i], label=prop['label'])
        axes[i].set_xlabel('UMAP Dimension 1')
        axes[i].set_ylabel('UMAP Dimension 2')
        #axes[i].set_title(f'Colored by {prop["label"]}')
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'umap_physical_properties_detailed.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Create individual high-resolution plots for each property
    for prop_name, prop in properties.items():
        plt.figure(figsize=(12, 10))

        scatter = plt.scatter(embedding[:, 0], embedding[:, 1],
                           c=prop['data'],
                           cmap=prop['cmap'],
                           s=5, alpha=0.8,
                           vmin=prop['vmin'],
                           vmax=prop['vmax'])

        plt.colorbar(scatter, label=prop['label'], shrink=0.8)
        plt.xlabel('UMAP Dimension 1')
        plt.ylabel('UMAP Dimension 2')
        #plt.title(f'UMAP Latent Space Colored by {prop["label"]}\n'
        #         f'({len(embedding)} galaxies)', fontsize=16)
        plt.xticks([])
        plt.yticks([])

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'umap_{prop_name}_colored.png'), dpi=300, bbox_inches='tight')
        plt.close()

    print(f"UMAP physical property plots saved for {len(embedding)} galaxies")

def load_cvae_model(model_path):
    """Load a specific CVAE model with the specified parameters"""
    params = {
        'd': 5,
        'lr_vae': 1.e-5,
        'lr_phys': 1.e-5,
        'batch_size': 512,
        'checkpoint_every_n_epochs': 5,
        'max_epochs': 1000,
        'max_num_samples': None,
        'max_num_samples_val': None,
        'pin_memory': True,
        'num_workers': 1,
        'ckpt_path': model_path,
        'mean_dict': {'full_means':1, 'full_stds':1},
        'beta0': np.ones(500),
        'beta1': np.ones(500),
        'beta2': np.ones(500),
        'segment': True,
        'lambda_roi': 0.9,
    }

    prefix = f"{params['d']}feats_focalflroi_wgtbright"
    params['logfile_name'] = f'cvae_ssl_2M_train_lr{params["lr_vae"]:.2e}_crop_{prefix}.log'

    print(f"Loading CVAE model from {model_path}...")
    cvae = LitCVAE(params=params)
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    cvae.load_state_dict(checkpoint["state_dict"])
    print(f"Successfully loaded model from {model_path}")
    return cvae

def detect_anomalies_reconstruction(model, save_dir, n_sigma=3):
    """
    Detect anomalies based on reconstruction error using multiple methods:
    1. Z-score of reconstruction error
    2. Local outlier factor in reconstruction error space
    3. Isolation forest on combined latent + reconstruction space
    """
    try:
        # Use only bright galaxies for consistency
        x_bright = model['images'][:]
        xhat_bright = model['recon'][:]

        #latent = model['latent'][:].reshape(NDIM, -1).T
        mu = model['latent'][:]
        latent = mu.reshape(len(mu), -1)

        params = np.squeeze(model['y_true'][:])

        if len(params) > 50000:
            idx = np.arange(len(latent))
            chosen = np.random.choice(idx, size=50000)
            latent = latent[chosen, :]
            params = params[chosen, :]
            x_bright = x_bright[chosen, :, :, :]
            xhat_bright = xhat_bright[chosen, :, :, :]

        # Calculate MSE for bright galaxies - need to handle the repeated images in xhat_bright
        mse = np.zeros(len(x_bright))
        for i in range(len(x_bright)):
            orig = x_bright[i]
            recon = xhat_bright[i]
            recon = recon[0]
            mse[i] = np.mean((orig - recon)**2)

        # Method 1: Z-score based detection
        zscore = stats.zscore(mse)
        zscore_outliers = np.where(np.abs(zscore) > n_sigma)[0]

        # Method 2: Local Outlier Factor
        lof = LocalOutlierFactor(contamination=0.1, n_neighbors=20)
        lof_labels = lof.fit_predict(mse.reshape(-1, 1))
        lof_outliers = np.where(lof_labels == -1)[0]

        # Method 3: Isolation Forest on combined features
        combined_features = np.hstack([latent, mse.reshape(-1, 1)])
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        iso_labels = iso_forest.fit_predict(combined_features)
        iso_outliers = np.where(iso_labels == -1)[0]

        # Plot the most anomalous galaxies from each method
        methods = {
            'zscore': zscore_outliers,
            'lof': lof_outliers,
            'isolation_forest': iso_outliers
        }

        for method_name, outlier_indices in methods.items():
            if len(outlier_indices) == 0:
                print(f"No anomalies found using {method_name}")
                continue

            # Sort outliers by their anomaly score
            if method_name == 'zscore':
                scores = np.abs(zscore[outlier_indices])
            elif method_name == 'lof':
                scores = -lof.negative_outlier_factor_[outlier_indices]
            else:
                scores = -iso_forest.score_samples(combined_features[outlier_indices])

            # Get top N most anomalous
            n_examples = min(5, len(outlier_indices))
            top_indices = outlier_indices[np.argsort(scores)[-n_examples:]]

            fig, axes = plt.subplots(3, n_examples, figsize=(4*n_examples, 12))
            plt.suptitle(f'Most Anomalous Galaxies ({method_name.replace("_", " ").title()})')#, y=1.02)

            for i, idx in enumerate(top_indices):
                # Original image
                orig = np.moveaxis(x_bright[idx], 0, -1)
                axes[0, i].imshow(orig)
                axes[0, i].set_title(f'Orig\nz={params[idx,0]:.2f}\nlog(M*)={params[idx,1]:.2f}')

                # Reconstruction
                recon = np.array(xhat_bright[idx], dtype=np.float32)
                # Handle the case where recon has shape (batch, redundant, channel, px_x, px_y) - take first redundant image
                recon = recon[:, 0, :, :, :]  # Take first redundant image
                recon = np.moveaxis(recon, 1, -1)  # Move channels to last dimension
                axes[1, i].imshow(recon)
                axes[1, i].set_title(f'Recon\nMSE={mse[idx]:.2e}')

                # Difference
                diff = np.abs(recon - orig)
                axes[2, i].imshow(diff, cmap='hot')
                axes[2, i].set_title('|Diff|')

                for ax in axes[:, i]:
                    ax.axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'anomalies_{method_name}.png'), dpi=300, bbox_inches='tight')
            plt.close()

    except Exception as e:
        print(f"Warning: Could not perform anomaly detection due to {str(e)}")
        return

def analyze_latent_clusters(model, save_dir):
    """
    Analyze clusters in latent space to identify different galaxy populations:
    1. Use UMAP for dimensionality reduction
    2. Apply HDBSCAN for clustering
    3. Visualize cluster representatives and their properties
    """
    #latent = model['latent'][:].reshape(NDIM, -1).T
    latent = model['latent'][:]
    latent = latent.reshape(len(latent), -1)

    params = np.squeeze(model['y_true'][:])
    x = model['images'][:]
    recon = model['recon'][:]

    if len(latent) > 50000:
        idx = np.arange(len(latent))
        chosen = np.random.choice(idx, size=50000)
        latent = latent[chosen, :]
        params = params[chosen, :]
        x = x[chosen, :, :, :]
        recon = recon[chosen, :, :, :]

    print(f"Original data shapes: latent {latent.shape}, params {params.shape}, images {x.shape}")

    # Store original indices for mapping back
    original_indices = np.arange(len(latent))

    # Reduce dimensionality with UMAP
    umap_reducer = UMAP(n_neighbors=5000, min_dist=1, n_components=2, random_state=42)
    embedding = umap_reducer.fit_transform(latent)

    min_cluster_size = max(100, len(latent) // 100)  # At least 1% of data per cluster
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=1000, cluster_selection_epsilon=0.5)
    cluster_labels = clusterer.fit_predict(embedding)

    # Count actual clusters (excluding noise with label -1)
    unique_labels = np.unique(cluster_labels)
    n_clusters = len(unique_labels[unique_labels >= 0])  # Exclude noise
    n_noise = np.sum(cluster_labels == -1)

    print(f"HDBSCAN parameters: min_cluster_size={min_cluster_size}")
    print(f"Found {n_clusters} clusters + {n_noise} noise points")
    print(f"Cluster counts: {np.bincount(cluster_labels[cluster_labels >= 0])}")

    # Plot UMAP embedding colored by clusters
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=cluster_labels, cmap='tab20', s=1, alpha=0.5)
    plt.colorbar(scatter)
    plt.title('UMAP Embedding\nColored by HDBSCAN Clusters')
    plt.savefig(os.path.join(save_dir, 'latent_clusters_umap.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # For each cluster, show representative galaxies and analyze properties
    unique_clusters = np.unique(cluster_labels)
    # Remove noise cluster (-1) and get only actual clusters
    actual_clusters = unique_clusters[unique_clusters >= 0]

    if len(actual_clusters) == 0:
        print("No clear clusters found")
        return

    n_examples = 5
    fig, axes = plt.subplots(len(actual_clusters), n_examples,
                            figsize=(15, 3*len(actual_clusters)))

    # Handle case of single cluster
    if len(actual_clusters) == 1:
        axes = axes.reshape(1, -1)

    for i, cluster in enumerate(actual_clusters):
        cluster_mask = cluster_labels == cluster
        cluster_points = latent[cluster_mask]
        cluster_params = params[cluster_mask]
        cluster_images = x[cluster_mask]  # Get images for this cluster
        cluster_recon = recon[cluster_mask]

        print(f"Cluster {cluster}: {np.sum(cluster_mask)} galaxies")

        # Calculate cluster center and find nearest examples
        center = np.mean(cluster_points, axis=0)
        distances = np.linalg.norm(cluster_points - center, axis=1)
        nearest_indices = np.argsort(distances)[:n_examples]

        # Plot cluster statistics
        #ax = axes[i, 0]
        #ax.hist(cluster_params[:, 0], bins=20, alpha=0.5, label='z')
        #ax.hist(cluster_params[:, 1], bins=20, alpha=0.5, label='log(M*)')
        #ax.set_title(f'Cluster {cluster}\nN={np.sum(cluster_mask)}')
        #ax.legend()

        # Plot representative galaxies - use direct indexing within cluster
        for j, idx in enumerate(nearest_indices):
            ax = axes[i, j]  # Changed from j+1 to j since we removed the first column
            img = np.moveaxis(cluster_images[idx], 0, -1)  # Use cluster_images[idx] directly
            #img = np.moveaxis(cluster_recon[idx], 0, -1)  # Use cluster_images[idx] directly
            ax.imshow(img)
            ax.axis('off')
            # Add some debug info
            ax.set_title(f'Rep {j+1}\nz={cluster_params[idx,0]:.3f}\nM*={cluster_params[idx,1]:.2f}')#, fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'cluster_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

def analyze_morphological_transitions(model, cvae, save_dir):
    """
    Analyze smooth transitions between different galaxy morphologies:
    1. Find pairs of galaxies with different morphologies
    2. Interpolate between them in latent space
    3. Show the transition of galaxy appearances
    """
    # Use only bright galaxies for consistency
    #mu = model['bright']['latent'][:].reshape(NDIM, -1).T
    mu = model['bright']['latent'][:]
    mu = mu.reshape(len(mu), -1)

    # Find some diverse pairs (using largest distances in latent space)
    n_samples = len(mu)
    n_pairs = 5
    pairs = []

    # Simple diversity sampling
    for _ in range(n_pairs):
        idx1 = np.random.randint(n_samples)

        # Find the most different galaxy in latent space
        distances = np.linalg.norm(mu - mu[idx1], axis=1)
        idx2 = np.argmax(distances)
        pairs.append((idx1, idx2))

    # For each pair, create interpolation
    n_steps = 8
    fig = plt.figure(figsize=(20, 4*n_pairs))

    for pair_idx, (idx1, idx2) in enumerate(pairs):
        z1 = torch.tensor(mu[idx1])
        z2 = torch.tensor(mu[idx2])

        # Create interpolation steps
        alphas = np.linspace(0, 1, n_steps)

        for step_idx, alpha in enumerate(alphas):
            # Interpolate in latent space
            z_interp = z1 * (1-alpha) + z2 * alpha

            # Decode
            with torch.no_grad():
                decoded = cvae.decode(z_interp.unsqueeze(0).to(cvae.device))
                img = decoded.squeeze(0).cpu().permute(1, 2, 0).numpy()

                # Plot
                ax = plt.subplot(n_pairs, n_steps, pair_idx*n_steps + step_idx + 1)
                ax.imshow(img)
                ax.axis('off')

                if step_idx == 0:
                    ax.set_title('Start')
                elif step_idx == n_steps-1:
                    ax.set_title('End')
                else:
                    ax.set_title(f'{alpha:.1f}')

    plt.suptitle('Latent Space (Gal-Gal) Traversals')#, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'morphological_transitions.png'), dpi=300, bbox_inches='tight')
    plt.close()

def analyze_nearest_neighbors(model, save_dir, n_query_galaxies=3, n_neighbors=5):
    """
    Find nearest neighbors in latent space for randomly selected galaxies

    Args:
        model: Dictionary containing model data
        save_dir: Directory to save plots
        n_query_galaxies: Number of random galaxies to use as queries
        n_neighbors: Number of nearest neighbors to find for each query
    """
    from sklearn.neighbors import NearestNeighbors

    mu = model['latent'][:]
    mu = mu.reshape(len(mu), -1)

    images = model['images'][:]
    params = np.squeeze(model['y_true'][:])

    n_samples = len(mu)
    print(f"Analyzing nearest neighbors in {mu.shape[1]}D latent space with {n_samples} galaxies")

    # Check for duplicate latent vectors
    unique_vectors, counts = np.unique(mu, axis=0, return_counts=True)
    print(f"Unique latent vectors: {len(unique_vectors)} out of {n_samples} total")
    if len(unique_vectors) < n_samples:
        print(f"WARNING: {n_samples - len(unique_vectors)} duplicate latent vectors found!")
        print(f"Most common vector appears {counts.max()} times")

    # Analyze latent space statistics
    print(f"Latent space statistics:")
    for i in range(mu.shape[1]):
        print(f"  Dim {i+1}: mean={mu[:,i].mean():.4f}, std={mu[:,i].std():.4f}, range=({mu[:,i].min():.4f}, {mu[:,i].max():.4f})")

    # Check if we should normalize the latent space
    dim_stds = mu.std(axis=0)
    max_std_ratio = dim_stds.max() / dim_stds.min()
    print(f"Standard deviation ratio (max/min): {max_std_ratio:.2f}")

    if max_std_ratio > 5:
        print("WARNING: Large variance differences between dimensions detected!")
        print("Consider using normalized latent space or alternative distance metrics")

    # Fit nearest neighbors model with different distance metrics
    # Start with Euclidean (default)
    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1, algorithm='auto', metric='euclidean')
    nbrs.fit(mu)

    # Alternative: fit with cosine similarity for comparison
    nbrs_cosine = NearestNeighbors(n_neighbors=n_neighbors+1, algorithm='auto', metric='cosine')
    nbrs_cosine.fit(mu)

    # Randomly select query galaxies
    #np.random.seed(32)  # For reproducibility
    query_indices = np.random.choice(n_samples, n_query_galaxies, replace=False)

    # Create figure
    fig, axes = plt.subplots(n_query_galaxies, n_neighbors+1, figsize=(4*(n_neighbors+1), 4*n_query_galaxies))
    if n_query_galaxies == 1:
        axes = axes.reshape(1, -1)

    for i, query_idx in enumerate(query_indices):
        # Find nearest neighbors using Euclidean distance
        distances_eucl, neighbor_indices_eucl = nbrs.kneighbors(mu[query_idx].reshape(1, -1))
        neighbor_indices_eucl = neighbor_indices_eucl[0][1:]  # Remove query itself
        distances_eucl = distances_eucl[0][1:]

        # Find nearest neighbors using cosine similarity
        distances_cos, neighbor_indices_cos = nbrs_cosine.kneighbors(mu[query_idx].reshape(1, -1))
        neighbor_indices_cos = neighbor_indices_cos[0][1:]  # Remove query itself
        distances_cos = distances_cos[0][1:]

        # Use Euclidean distance for display (better for finding visually similar galaxies)
        neighbor_indices = neighbor_indices_eucl
        distances = distances_eucl



        # Plot query galaxy
        query_img = np.moveaxis(images[query_idx], 0, -1)
        axes[i, 0].imshow(query_img)
        axes[i, 0].set_title(f'Query {i+1}', fontweight='bold') #\nz={params[query_idx,0]:.3f}\nlog(M*)={params[query_idx,1]:.2f}',
                                                                #fontweight='bold')
        axes[i, 0].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        # Add red border to query
        for spine in axes[i, 0].spines.values():
            spine.set_edgecolor('red')
            spine.set_linewidth(3)

        # Plot nearest neighbors
        for j, (neighbor_idx, distance) in enumerate(zip(neighbor_indices, distances)):
            neighbor_img = np.moveaxis(images[neighbor_idx], 0, -1)
            axes[i, j+1].imshow(neighbor_img)
            axes[i, j+1].set_title(f'NN {j+1}\nd={distance:.3f}') #z={params[neighbor_idx,0]:.3f}\nlog(M*)={params[neighbor_idx,1]:.2f}')
            axes[i, j+1].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    plt.suptitle(f'Nearest Neighbors (Euclidean distance)')#, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'nearest_neighbors_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Print detailed statistics comparing both distance metrics
    print(f"\nNearest Neighbor Analysis Results (Comparing Distance Metrics):")
    for i, query_idx in enumerate(query_indices):
        # Get results for both metrics
        distances_eucl, neighbor_indices_eucl = nbrs.kneighbors(mu[query_idx].reshape(1, -1))
        neighbor_indices_eucl = neighbor_indices_eucl[0][1:]
        distances_eucl = distances_eucl[0][1:]

        distances_cos, neighbor_indices_cos = nbrs_cosine.kneighbors(mu[query_idx].reshape(1, -1))
        neighbor_indices_cos = neighbor_indices_cos[0][1:]
        distances_cos = distances_cos[0][1:]

        print(f"\nQuery Galaxy {i+1} (Index {query_idx}):")
        print(f"  Query properties: z={params[query_idx,0]:.4f}, log(M*)={params[query_idx,1]:.3f}")

        # Compare overlap between the two methods
        eucl_set = set(neighbor_indices_eucl)
        cos_set = set(neighbor_indices_cos)
        overlap = len(eucl_set.intersection(cos_set))
        print(f"  Overlap between Euclidean and Cosine neighbors: {overlap}/{n_neighbors}")

        print(f"  Euclidean neighbors:")
        for j, (neighbor_idx, distance) in enumerate(zip(neighbor_indices_eucl, distances_eucl)):
            print(f"    NN {j+1}: Index {neighbor_idx}, Distance {distance:.4f}, z={params[neighbor_idx,0]:.4f}, log(M*)={params[neighbor_idx,1]:.3f}")

        print(f"  Cosine neighbors:")
        for j, (neighbor_idx, distance) in enumerate(zip(neighbor_indices_cos, distances_cos)):
            print(f"    NN {j+1}: Index {neighbor_idx}, Distance {distance:.4f}, z={params[neighbor_idx,0]:.4f}, log(M*)={params[neighbor_idx,1]:.3f}")

        # Calculate statistics for Euclidean neighbors
        neighbor_z_eucl = params[neighbor_indices_eucl, 0]
        neighbor_mass_eucl = params[neighbor_indices_eucl, 1]
        z_std_eucl = np.std(neighbor_z_eucl)
        mass_std_eucl = np.std(neighbor_mass_eucl)

        # Calculate statistics for Cosine neighbors
        neighbor_z_cos = params[neighbor_indices_cos, 0]
        neighbor_mass_cos = params[neighbor_indices_cos, 1]
        z_std_cos = np.std(neighbor_z_cos)
        mass_std_cos = np.std(neighbor_mass_cos)

        print(f"    Physical property consistency:")
        print(f"    Euclidean - Redshift std: {z_std_eucl:.4f}, Mass std: {mass_std_eucl:.3f}")
        print(f"    Cosine - Redshift std: {z_std_cos:.4f}, Mass std: {mass_std_cos:.3f}")

def plot_latent_corner_plot_subset(model, save_dir, max_points=50000):
    """
    Create pairwise scatter plots of latent dimensions colored by physical parameters

    Args:
        model: Dictionary containing model data
        save_dir: Directory to save plots
        max_points: Maximum number of points to plot (for performance)
    """
    # Get latent space and physical parameters
    mu = model['latent'][:]
    mu = mu.reshape(len(mu), -1)
    params = np.squeeze(model['y_true'][:])

    # Subsample if needed for performance
    if len(mu) > max_points:
        idx = np.random.choice(len(mu), max_points, replace=False)
        # Sort indices for h5py compatibility
        idx = np.sort(idx)
        mu_sample = mu[idx]
        params_sample = params[idx]
    else:
        mu_sample = mu
        params_sample = params

    n_dims = mu_sample.shape[1]

    # Define physical properties to color by
    properties = {
        'redshift': {
            'data': params_sample[:, 0],
            'cmap': 'viridis',
            'label': 'Redshift',
            'vmin': np.percentile(params_sample[:, 0], 5),
            'vmax': np.percentile(params_sample[:, 0], 95)
        },
        'mass': {
            'data': params_sample[:, 1],
            'cmap': 'magma',
            'label': r'log(M*/M_sun)',
            'vmin': np.percentile(params_sample[:, 1], 5),
            'vmax': np.percentile(params_sample[:, 1], 95)
        },
        'sfr': {
            'data': params_sample[:, 2],
            'cmap': 'inferno',
            'label': 'log(SFR/Gyr)',
            'vmin': -2,
            'vmax': np.percentile(params_sample[:, 2], 95)
        }
    }

    # Create plots for key latent dimension pairs
    dim_pairs = [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3), (0, 4), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (2, 8), (1, 9), (2, 9)]

    for prop_name, prop in properties.items():
        # Create a grid of subplots for this physical property
        n_pairs = len(dim_pairs)
        n_cols = 3
        n_rows = (n_pairs + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        fig.suptitle(f'Latent Dimensions Colored by {prop["label"]}')#, y=1.02)

        # Flatten axes for easier indexing
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes_flat = axes.flatten()

        for i, (dim1, dim2) in enumerate(dim_pairs):
            ax = axes_flat[i]

            scatter = ax.scatter(mu_sample[:, dim1], mu_sample[:, dim2],
                               c=prop['data'],
                               cmap=prop['cmap'],
                               s=3, alpha=0.6,
                               vmin=prop['vmin'],
                               vmax=prop['vmax'])

            ax.set_xlabel(f'Latent Dimension {dim1+1}')
            ax.set_ylabel(f'Latent Dimension {dim2+1}')
            ax.set_title(f'Dim {dim1+1} vs Dim {dim2+1}')

            # Add colorbar to each subplot
            plt.colorbar(scatter, ax=ax, label=prop['label'], shrink=0.8)

        # Hide unused subplots
        for j in range(len(dim_pairs), len(axes_flat)):
            axes_flat[j].set_visible(False)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'latent_pairs_{prop_name}_colored.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # Also create a comprehensive view with all three properties for the most important pair (0,1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    #fig.suptitle('First Two Latent Dimensions Colored by Physical Properties', fontsize=16)

    for i, (prop_name, prop) in enumerate(properties.items()):
        scatter = axes[i].scatter(mu_sample[:, 0], mu_sample[:, 1],
                                c=prop['data'],
                                cmap=prop['cmap'],
                                s=5, alpha=0.7,
                                vmin=prop['vmin'],
                                vmax=prop['vmax'])

        axes[i].set_xlabel('Latent Dimension 1')
        axes[i].set_ylabel('Latent Dimension 2')
        axes[i].set_title(f'Colored by {prop["label"]}')
        plt.colorbar(scatter, ax=axes[i], label=prop['label'])

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'latent_dims_1_2_all_properties.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Latent dimension pairwise saved for {len(mu_sample)} galaxies")

def plot_latent_corner_plot(model, save_dir, max_points=50000):
    """
    Create a complete corner plot of all latent dimensions colored by physical parameters

    Args:
        model: Dictionary containing model data
        save_dir: Directory to save plots
        max_points: Maximum number of points to plot (for performance)
    """
    # Get latent space and physical parameters
    mu = model['latent'][:]
    mu = mu.reshape(len(mu), -1)
    params = np.squeeze(model['y_true'][:])

    # Subsample if needed for performance
    if len(mu) > max_points:
        idx = np.random.choice(len(mu), max_points, replace=False)
        # Sort indices for h5py compatibility
        idx = np.sort(idx)
        mu_sample = mu[idx]
        params_sample = params[idx]
    else:
        mu_sample = mu
        params_sample = params

    n_dims = mu_sample.shape[1]
    print(f"Creating corner plot for {n_dims} latent dimensions with {len(mu_sample)} galaxies")

    # Define physical properties to color by
    properties = {
        'redshift': {
            'data': params_sample[:, 0],
            'cmap': 'viridis',
            'label': 'Redshift',
            'vmin': np.percentile(params_sample[:, 0], 5),
            'vmax': np.percentile(params_sample[:, 0], 95)
        },
        'mass': {
            'data': params_sample[:, 1],
            'cmap': 'magma',
            'label': 'log($M*/M_{\odot}$)',
            'vmin': np.percentile(params_sample[:, 1], 5),
            'vmax': np.percentile(params_sample[:, 1], 95)
        },
        'sfr': {
            'data': params_sample[:, 2],
            'cmap': 'inferno',
            'label': 'log(SFR/Gyr)',
            'vmin': -2,
            'vmax': np.percentile(params_sample[:, 2], 95)
        }
    }

    # Create corner plots for each physical property
    for prop_name, prop in properties.items():
        print(f"Creating corner plot colored by {prop['label']}...")

        # Create the corner plot grid
        fig, axes = plt.subplots(n_dims, n_dims, figsize=(6*n_dims, 6*n_dims))
        fig.suptitle(f'Latent Dimensions Corner Plot Colored by {prop["label"]}')#, y=1.02)

        # Create all pairwise combinations
        for i in range(n_dims):
            for j in range(n_dims):
                ax = axes[i, j]

                if i == j:
                    # Diagonal: histogram of the dimension
                    ax.hist(mu_sample[:, i], bins=50, alpha=0.7, color='gray', density=True)
                    ax.set_xlabel(f'Dim {i+1}')
                    ax.set_ylabel('Density')
                    # No title needed for marginals
                else:
                    # Off-diagonal: scatter plot
                    scatter = ax.scatter(mu_sample[:, j], mu_sample[:, i],
                                       c=prop['data'],
                                       cmap=prop['cmap'],
                                       s=3, alpha=0.6,
                                       vmin=prop['vmin'],
                                       vmax=prop['vmax'])

                    ax.set_xlabel(f'Dim {j+1}')
                    ax.set_ylabel(f'Dim {i+1}')

                    # Add colorbar to the last subplot in each row
                    if j == n_dims - 1:
                        cbar = plt.colorbar(scatter, ax=ax, label=prop['label'], shrink=0.8)

        plt.tight_layout(pad=3.0)  # Add more padding to prevent overlap and ensure proper layout
        plt.savefig(os.path.join(save_dir, f'latent_corner_plot_{prop_name}.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()

    # Removed the problematic 3x3 grid section - the main corner plots are sufficient

    print(f"Corner plots saved for {len(mu_sample)} galaxies with {n_dims} dimensions")

def plot_latent_corner_plot_gz(save_dir, max_points=50000):
    """
    Create a complete corner plot of all latent dimensions colored by galaxy zoo correlations

    Args:
        model: Dictionary containing model data
        save_dir: Directory to save plots
        max_points: Maximum number of points to plot (for performance)
    """
    # Get latent space and physical parameters
    gz_file = h5py.File("/Users/alexgagliano/Documents/Research/GalaxyAutoencoder/data/test_subset/maskeddaep_results_gzsample_1752682352_maskdaep5_crossmatch.h5")

    mu = gz_file['from_file_a']['latent'][:]
    mu = mu.reshape(len(mu), -1)
    gz_feats = ['how-rounded_round_fraction', "has-spiral-arms_yes_fraction", "merging_major-disturbance_fraction", "petro_theta", "mag_r"]
    params = np.vstack([gz_file['from_file_b'][x][:] for x in gz_feats]).T
    print("shape params:", params.shape)

    # Subsample if needed for performance
    if len(mu) > max_points:
        idx = np.random.choice(len(mu), max_points, replace=False)
        # Sort indices for h5py compatibility
        idx = np.sort(idx)
        mu_sample = mu[idx]
        params_sample = params[idx]
    else:
        mu_sample = mu
        params_sample = params

    n_dims = mu_sample.shape[1]
    print(f"Creating corner plot for {n_dims} latent dimensions with {len(mu_sample)} galaxies")

    # Define physical properties to color by
    properties = {

        gz_feats[0]: {
            'data': params_sample[:, 0],
            'cmap': 'viridis',
            'label': 'Rounded Fraction',
            'vmin': np.percentile(params_sample[:, 0], 5),
            'vmax': np.percentile(params_sample[:, 0], 95)
        },
        gz_feats[1]: {
            'data': params_sample[:, 1],
            'cmap': 'magma',
            'label': 'Spiral Arms Fraction',
            'vmin': np.percentile(params_sample[:, 1], 5),
            'vmax': np.percentile(params_sample[:, 1], 95)
        },
        gz_feats[2]: {
            'data': params_sample[:, 2],
            'cmap': 'inferno',
            'label': 'Major Merger Fraction',
            'vmin': np.percentile(params_sample[:, 2], 5),
            'vmax': np.percentile(params_sample[:, 2], 95)
        },
        gz_feats[3]: {
            'data': params_sample[:, 3],
            'cmap': 'summer',
            'label': 'r-Band Petrosian Radius',
            'vmin': np.percentile(params_sample[:, 3], 5),
            'vmax': np.percentile(params_sample[:, 3], 95)
        },
        gz_feats[4]: {
            'data': params_sample[:, 4],
            'cmap': 'BrBG',
            'label': 'r-Band Magnitude',
            'vmin': 17, #np.percentile(params_sample[:, 4], 5),
            'vmax': 18, #np.percentile(params_sample[:, 4], 95)
        },
    }

    # Create corner plots for each physical property
    for prop_name, prop in properties.items():
        print(f"Creating corner plot colored by {prop['label']}...")

        # Create the corner plot grid
        fig, axes = plt.subplots(n_dims, n_dims, figsize=(6*n_dims, 6*n_dims))
        fig.suptitle(f'Latent Dimensions Corner Plot Colored by {prop["label"]}')#, y=1.02)

        # Create all pairwise combinations
        for i in range(n_dims):
            for j in range(n_dims):
                ax = axes[i, j]

                if i == j:
                    # Diagonal: histogram of the dimension
                    ax.hist(mu_sample[:, i], bins=50, alpha=0.7, color='gray', density=True)
                    ax.set_xlabel(f'Dim {i+1}')
                    ax.set_ylabel('Density')
                    # No title needed for marginals
                else:
                    # Off-diagonal: scatter plot
                    scatter = ax.scatter(mu_sample[:, j], mu_sample[:, i],
                                       c=prop['data'],
                                       cmap=prop['cmap'],
                                       s=3, alpha=0.6,
                                       vmin=prop['vmin'],
                                       vmax=prop['vmax'])

                    ax.set_xlabel(f'Dim {j+1}')
                    ax.set_ylabel(f'Dim {i+1}')

                    # Add colorbar to the last subplot in each row
                    if j == n_dims - 1:
                        cbar = plt.colorbar(scatter, ax=ax, label=prop['label'], shrink=0.8)

        plt.tight_layout(pad=3.0)  # Add more padding to prevent overlap and ensure proper layout
        plt.savefig(os.path.join(save_dir, f'latent_corner_plot_{prop_name}.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()

    # Removed the problematic 3x3 grid section - the main corner plots are sufficient

    print(f"Corner plots saved for {len(mu_sample)} galaxies with {n_dims} dimensions")



def plot_latent_corner_thumbnails(model, save_dir, max_points=50000, n_samples=800, thumbnail_size=0.18):
    """
    Create corner plots with galaxy thumbnails in each subplot

    Args:
        model: Dictionary containing model data
        save_dir: Directory to save plots
        max_points: Maximum number of points to plot (for performance)
        n_samples: Number of thumbnail samples to show per plot
        thumbnail_size: Size of thumbnails relative to figure
    """
    # Get latent space and images
    mu = model['latent'][:]
    mu = mu.reshape(len(mu), -1)

    # Subsample if needed for performance
    if len(mu) > max_points:
        idx = np.random.choice(len(mu), max_points, replace=False)
        # Sort indices for h5py compatibility
        idx = np.sort(idx)
        mu_sample = mu[idx]
        images_sample = model['images'][idx]
    else:
        mu_sample = mu
        images_sample = model['images'][:]

    n_dims = mu_sample.shape[1]
    print(f"Creating corner thumbnail plots for {n_dims} latent dimensions with {len(mu_sample)} galaxies")

    # Create the corner plot grid
    fig, axes = plt.subplots(n_dims, n_dims, figsize=(6*n_dims, 6*n_dims))
    fig.suptitle('Latent Dimensions Corner Plot with Galaxy Thumbnails')#, y=1.02)

    # Create all pairwise combinations
    for i in range(n_dims):
        for j in range(n_dims):
            ax = axes[i, j]

            if i == j:
                # Diagonal: histogram of the dimension
                ax.hist(mu_sample[:, i], bins=50, alpha=0.7, color='gray', density=True)
                ax.set_xlabel(f'Dim {i+1}')
                ax.set_ylabel('Density')
                # No title needed for marginals
            else:
                # Off-diagonal: scatter plot with thumbnails
                z1 = mu_sample[:, j]  # x-axis
                z2 = mu_sample[:, i]  # y-axis

                # Plot background points for context
                ax.scatter(z1, z2, c='lightgray', s=1, alpha=0.3)

                # Calculate axis limits and add padding
                x_min, x_max = z1.min(), z1.max()
                y_min, y_max = z2.min(), z2.max()

                x_padding = 0.05 * (x_max - x_min)
                y_padding = 0.05 * (y_max - y_min)

                ax.set_xlim(x_min - x_padding, x_max + x_padding)
                ax.set_ylim(y_min - y_padding, y_max + y_padding)

                # Sample points for thumbnails using grid-based approach
                n_gridx = int(np.sqrt(n_samples))  # Use full sample count for grid
                n_gridy = int(np.sqrt(n_samples))

                x_bins = np.linspace(x_min, x_max, n_gridx + 1)
                y_bins = np.linspace(y_min, y_max, n_gridy + 1)

                thumbnails_placed = 0
                placed_positions = []  # Track placed thumbnail positions
                min_distance = 0.03 * (x_max - x_min)  # Very small minimum distance

                for k in range(n_gridx):
                    for l in range(n_gridy):
                        # Find points in this grid cell
                        in_cell = (z1 >= x_bins[k]) & (z1 < x_bins[k+1]) & \
                                  (z2 >= y_bins[l]) & (z2 < y_bins[l+1])

                        if np.sum(in_cell) > 0:
                            # Pick a random point from this cell
                            cell_indices = np.where(in_cell)[0]
                            idx = np.random.choice(cell_indices)

                            # Check if this position is too close to already placed thumbnails
                            current_pos = (z1[idx], z2[idx])
                            too_close = False
                            for placed_pos in placed_positions:
                                if np.sqrt((current_pos[0] - placed_pos[0])**2 +
                                         (current_pos[1] - placed_pos[1])**2) < min_distance:
                                    too_close = True
                                    break

                            if not too_close:
                                try:
                                    # Get galaxy image
                                    img = np.array(images_sample[idx], dtype=np.float32)
                                    img = np.moveaxis(img, 0, -1)

                                    # Create offsetbox
                                    imagebox = OffsetImage(img, zoom=thumbnail_size)

                                    # Position the thumbnail
                                    ab = AnnotationBbox(imagebox, (z1[idx], z2[idx]),
                                                      frameon=True, pad=0.0,
                                                      bboxprops=dict(edgecolor='black', linewidth=0.5))

                                    # Add to plot
                                    ax.add_artist(ab)
                                    thumbnails_placed += 1
                                    placed_positions.append(current_pos)
                                except Exception as e:
                                    print(f"Error placing thumbnail: {e}")

                ax.set_xlabel(f'Dim {j+1}')
                ax.set_ylabel(f'Dim {i+1}')

    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.4, hspace=0.4)
    plt.savefig(os.path.join(save_dir, 'latent_corner_thumbnails.png'),
               dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Corner thumbnail plot saved for {len(mu_sample)} galaxies with {n_dims} dimensions")

def plot_umap_grid_thumbnails(data, save_path, grid_rows=60, grid_cols=200, image_size=48):
    """
    Create a widescreen tight grid of thumbnails approximating the UMAP layout.
    Uses optimal transport to assign UMAP coordinates to grid positions.

    Args:
        data: dict with 'latent' and 'images' keys
        save_path: directory where 'umap_model_grid.png' will be saved
        grid_rows: number of rows in the output grid
        grid_cols: number of columns in the output grid
        image_size: size of each thumbnail in pixels
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.manifold import SpectralEmbedding
    from scipy.optimize import linear_sum_assignment
    from PIL import Image
    from umap import UMAP

    latent = data['latent'][:].reshape(len(data['latent']), -1)
    images = data['images'][:]
    if images.ndim == 4 and images.shape[1] in (1, 3):
        images = np.moveaxis(images, 1, -1)

    # UMAP on all data
    reducer = UMAP(n_components=2, random_state=42)
    umap_coords = reducer.fit_transform(latent)
    umap_coords -= umap_coords.min(0)
    umap_coords /= umap_coords.max(0)

    n_cells = grid_rows * grid_cols
    if len(umap_coords) < n_cells:
        raise ValueError(f"Need at least {n_cells} samples to fill the grid.")

    # Randomly select representative points to preserve spread
    rng = np.random.default_rng(42)
    selected_idx = rng.choice(len(umap_coords), size=n_cells, replace=False)
    umap_subset = umap_coords[selected_idx]
    image_subset = images[selected_idx]

    # Grid coordinates
    grid_x, grid_y = np.meshgrid(np.linspace(0, 1, grid_cols), np.linspace(0, 1, grid_rows))
    grid_coords = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)

    # Solve optimal assignment
    cost_matrix = np.linalg.norm(umap_subset[:, None, :] - grid_coords[None, :, :], axis=2)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Create canvas
    canvas = np.ones((grid_rows * image_size, grid_cols * image_size, 3), dtype=np.uint8) * 0

    for idx_data, idx_grid in zip(row_ind, col_ind):
        gx = idx_grid % grid_cols
        gy = idx_grid // grid_cols
        top = gy * image_size
        left = gx * image_size

        img = image_subset[idx_data]
        if img.dtype != np.uint8:
            img = (img * 255).clip(0, 255).astype(np.uint8)
        thumb = Image.fromarray(img).resize((image_size, image_size))

        canvas[top:top+image_size, left:left+image_size] = np.array(thumb)

    Image.fromarray(canvas).save(f"{save_path}/umap_model_grid.png")



def plot_umap_grid_thumbnails_retry(data, save_path, grid_shape=(50, 100), image_size=64):
    """
    Create a tight, widescreen grid of thumbnails approximating UMAP layout with density preservation.

    Args:
        data: dict with 'latent' and 'images' keys
        save_path: directory where 'umap_model_grid.png' will be saved
        grid_shape: (rows, cols) of output grid
        image_size: size of each thumbnail in pixels
    """
    latent = data['latent'][:].reshape(len(data['latent']), -1)
    images = data['images'][:]
    if images.ndim == 4 and images.shape[1] in (1, 3):
        images = np.moveaxis(images, 1, -1)

    # Fit UMAP
    reducer = UMAP(n_components=2, random_state=42)
    umap_coords = reducer.fit_transform(latent)
    umap_coords -= umap_coords.min(0)
    umap_coords /= umap_coords.max(0)

    # Normalize coordinates to [0,1] and store KDTree
    assigned = set()
    kdtree = KDTree(umap_coords)

    n_rows, n_cols = grid_shape
    n_cells = n_rows * n_cols
    canvas = np.zeros((n_rows * image_size, n_cols * image_size, 3), dtype=np.uint8)

    # Generate grid positions (in normalized [0,1] space)
    grid_x, grid_y = np.meshgrid(
        np.linspace(0, 1, n_cols),
        np.linspace(0, 1, n_rows)
    )
    grid_points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)

    used_indices = set()
    for i, gp in enumerate(grid_points):
        dist, idx = kdtree.query([gp], k=10)  # Look at 10 nearest neighbors
        for candidate in idx[0]:
            if candidate not in used_indices:
                used_indices.add(candidate)
                gx = i % n_cols
                gy = i // n_cols
                top = gy * image_size
                left = gx * image_size

                img = images[candidate]
                if img.dtype != np.uint8:
                    img = (img * 255).clip(0, 255).astype(np.uint8)
                thumb = Image.fromarray(img).resize((image_size, image_size))
                canvas[top:top + image_size, left:left + image_size] = np.array(thumb)
                break  # stop after assigning one valid candidate

    os.makedirs(save_path, exist_ok=True)
    Image.fromarray(canvas).save(os.path.join(save_path, 'umap_model_grid_dens.png'))

def process_single_model(model_name, model_paths, plots_dir):
    """Process a single model file"""
    print(f"Processing {model_name}...")
    model_dir = os.path.join(plots_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    #try:
    if True:
        # Load the specific CVAE model for this test set
        daep_loaded = False
        try:
            cvae = load_cvae_model(model_paths['checkpoint'])
            daep_loaded = True
        except Exception as e:
            print(f"Warning: Could not load CVAE model: {str(e)}")
            print("Continuing with analyses that don't require the model...")

        #if True:
        with h5py.File(model_paths['test_results'], 'r') as model:
            # Generate all plots
            #shown_idxs = plot_reconstruction_comparison(
            #    model, model_dir,
            #        f"{model_name} CVAE, {model_name.split('_')[0]} features"
            #    )

            #plot_umap(model, model_dir)
            #plot_latent_corner_plot(model, model_dir)
            #plot_latent_corner_plot_gz(model_dir)
            #plot_latent_corner_thumbnails(model, model_dir, n_samples=800, thumbnail_size=0.18)

            # Create latent traversal plot - requires CVAE model
            #if daep_loaded:
            #    try:
            #        plot_latent_dimension_traversal(model, cvae, model_dir)
            #    except Exception as e:
            #        print(f"Warning: Could not generate latent traversal: {str(e)}")

            # Create latent space visualization with thumbnails
            #print("Generating latent space visualizations with thumbnails...")

            #plot_galaxies_in_umap_space(model, model_dir, n_samples=3000, thumbnail_size=0.3)
            #plot_galaxies_in_latent_space(model, model_dir, n_samples=2000, thumbnail_size=0.4)

            # Additional analysis
            #try:
            #    detect_anomalies_reconstruction(model, model_dir)
            #except Exception as e:
            #    print(f"Warning: Could not perform anomaly detection: {str(e)}")

            #try:
            #    analyze_latent_clusters(model, model_dir)
            #except Exception as e:
            #    print(f"Warning: Could not perform cluster analysis: {str(e)}")

            # Morphological transitions require CVAE model
            #if daep_loaded:
            #    try:
            #        analyze_morphological_transitions(model, cvae, model_dir)
            #    except Exception as e:
            #        print(f"Warning: Could not analyze morphological transitions: {str(e)}")

            # Nearest neighbors analysis - doesn't require CVAE model
            #try:
            #    analyze_nearest_neighbors(model, model_dir)
            #except Exception as e:
            #    print(f"Warning: Could not perform nearest neighbor analysis: {str(e)}")

            plot_umap_grid_thumbnails(model, model_dir)

        # Clean up CVAE model
        if daep_loaded:
            del cvae
            gc.collect()

    print(f"Finished processing {model_name}")
    gc.collect()

def main():
    # Set up directories
    base_dir = "/Users/alexgagliano/Documents/Research/GalaxyAutoencoder"
    plots_dir = os.path.join(base_dir, "plots/validation")
    os.makedirs(plots_dir, exist_ok=True)

    # Get model paths
    model_paths = load_models(base_dir)

    # Process each model one at a time
    for model_name, paths in model_paths.items():
        process_single_model(model_name, paths, plots_dir)

if __name__ == "__main__":
    main()
