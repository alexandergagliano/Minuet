#!/bin/bash
#SBATCH -t 3-00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=16
#SBATCH --account=iaifi_lab
#SBATCH -p iaifi_gpu_priority
#SBATCH -n 1
#SBATCH -o myoutput_maskeddaep_5_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e myerrors_maskeddaep_5_%j.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH -J maskeddaep5
#SBATCH --mem=0       # Memory pool for all cores (see also --mem-per-cpu)

# load modules
date
source ~/.bashrc
conda activate tf2.12_cuda11_sbi
export LD_LIBRARY_PATH=/n/home02/agagliano/.conda/envs/tf2.12_cuda11_sbi/lib
srun python ../scripts/train_maskedminuet.py 2.e-4
date
