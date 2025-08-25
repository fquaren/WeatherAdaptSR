#!/bin/bash -l
#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type ALL
#SBATCH --mail-user filippo.quarenghi@unil.ch

#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name cluster
#SBATCH --output outputs/%j
#SBATCH --error job_errors/%j

#SBATCH --partition gpu-gh
#SBATCH --gres gpu:0
#SBATCH --gres-flags enforce-binding
#SBATCH --cpus-per-task 32
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 100G
#SBATCH --time 01:00:00

module load micromamba 
source /users/fquareng/.bashrc
micromamba -n dl run python /work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/WeatherAdaptSR/data/clustering_v4.py
