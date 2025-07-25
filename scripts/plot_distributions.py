import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
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

base_dir = '../split_files'
#splits = ['Train', 'Val', 'Test']
splits = ['Train', 'Test']
palette = sns.color_palette("Dark2", n_colors=3)
colors = dict(zip(splits, palette))  

z_data = {}
phot_data = {}
mass_data = {}

for split in splits:
    z_all = []
    mass_all = []
    phot_r_all = []

    split_dir = os.path.join(base_dir, split)
    h5_files = sorted([f for f in os.listdir(split_dir) if f.endswith('.hdf5')])

    for fname in h5_files:
        fpath = os.path.join(split_dir, fname)
        with h5py.File(fpath, 'r') as hfile:
            photo_z = hfile['PHOTO_Z'][:]
            photo_zerr = hfile['PHOTO_ZERR'][:]
            spec_z = hfile['SPEC_Z'][:]

            z = photo_z.copy()
            use_spec = spec_z > 0
            z[use_spec] = spec_z[use_spec]

            mass = hfile['MASS_BEST'][:]
            phot_r = hfile['MAG_R'][:] 

            z_all.append(z)
            phot_r_all.append(phot_r)
            mass_all.append(mass)

    z_data[split] = np.concatenate(z_all)
    mass_data[split] = np.concatenate(mass_all)
    phot_data[split] = np.concatenate(phot_r_all)

# Plotting Redshift Histogram
plt.figure(figsize=(7, 6))
for split in splits:
    plt.hist(z_data[split], bins=np.linspace(0, 1.0), alpha=0.5, label=split, color=colors[split], histtype='stepfilled', lw=3)
    print(f"Split: {split}")
    print(f"Number of galaxies: {len(z_data[split])}")
plt.xlabel(r"$z$")
plt.ylabel("Number of Galaxies")
plt.yscale("log")
plt.tight_layout()
plt.show()
plt.savefig("./distributions_z.png", dpi=300, bbox_inches='tight')

plt.figure(figsize=(7, 6))
for split in splits:
    plt.hist(phot_data[split], bins=np.linspace(16, 20), alpha=0.5, label=split, color=colors[split], histtype='stepfilled', lw=3)
plt.yscale("log")
plt.xlabel(r"$m_r$")
plt.ylabel("Number of Galaxies")
plt.tight_layout()
plt.savefig("./distributions_mag.png", dpi=300, bbox_inches='tight')

# Plotting Stellar Mass Histogram
plt.figure(figsize=(7, 6))
for split in splits:
    plt.hist(mass_data[split], bins=np.linspace(8, 12), alpha=0.5, label=split, color=colors[split], histtype='stepfilled', lw=3)
plt.xlabel(r"log$_{10}(M_*/M_{\odot})$")
plt.ylabel("Number of Galaxies")
plt.yscale("log")
plt.tight_layout()
plt.savefig("./distributions_mass.png", dpi=300, bbox_inches='tight')

