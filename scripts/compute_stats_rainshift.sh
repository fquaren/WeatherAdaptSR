#!/bin/bash -l
#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type ALL
#SBATCH --mail-user filippo.quarenghi@unil.ch

#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name prep
#SBATCH --output outputs/%j
#SBATCH --error job_errors/%j

#SBATCH --partition cpu
#SBATCH --nodes 1
#SBATCH --cpus-per-task 4
#SBATCH --mem 50G
#SBATCH --time 24:00:00

source /users/fquareng/.bashrc
micromamba activate dl-torch
micromamba -n dl-torch run python /work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/WeatherAdaptSR/data/compute_stats_rainshift.py