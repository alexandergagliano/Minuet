#!/bin/bash
#SBATCH -t 1-00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH --account=iaifi_lab
#SBATCH -p iaifi_gpu_priority
##SBATCH -p itc_gpu
#SBATCH -o myoutput_testmaskeddaep_%j.out  
#SBATCH -e myerrors_testmaskeddaep_%j.err  
#SBATCH -J test_maskeddaep
#SBATCH --mem=0       

# load modules
date
source ~/.bashrc
conda activate tf2.12_cuda11_sbi
cd /n/holylabs/LABS/iaifi_lab/Users/agagliano/galaxyAutoencoder/submit/
export LD_LIBRARY_PATH=/n/home02/agagliano/.conda/envs/tf2.12_cuda11_sbi/lib
python ../scripts/test_daep.py
#python ../scripts/test_galaxyzoo.py
date
