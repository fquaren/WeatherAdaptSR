#!/bin/bash -l
#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type ALL
#SBATCH --mail-user filippo.quarenghi@unil.ch

#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name analysis
#SBATCH --output outputs/%j
#SBATCH --error job_errors/%j

#SBATCH --partition cpu
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 12
#SBATCH --mem 350G
#SBATCH --time 12:00:00

source /users/fquareng/.bashrc
micromamba activate dl-torch
micromamba run -n dl-torch python /work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/WeatherAdaptSR/analysis.py