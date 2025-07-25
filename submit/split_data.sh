#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --job-name=splitData
#SBATCH -c 1                # Number of cores (-c)
#SBATCH -p shared 
#SBATCH --mem=0           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -n 1

# set up for problem & define any environment variables here
source ~/.bashrc
conda activate tf2.12_cuda11_sbi
export LD_LIBRARY_PATH=/n/home02/agagliano/.conda/envs/tf2.12_cuda11_sbi/lib
python /n/holylabs/LABS/iaifi_lab/Users/agagliano/galaxyAutoencoder/scripts/splitData.py
