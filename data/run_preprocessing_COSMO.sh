#!/bin/bash -l
#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type ALL
#SBATCH --mail-user filippo.quarenghi@unil.ch

#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name cosmo_zarr_prep
#SBATCH --output outputs/%j
#SBATCH --error job_errors/%j

#SBATCH --partition cpu
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 8
#SBATCH --mem 150G
#SBATCH --time 1:00:00

# Critical HPC Thread Management:
# xarray.open_mfdataset(parallel=True) uses Dask.
# We must restrict NumPy/OpenBLAS/MKL from spawning threads beyond the SLURM allocation
# to prevent catastrophic CPU thrashing on the shared node.
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

source /users/fquareng/.bashrc
micromamba activate dl-torch

# Assuming you saved the python pipeline as preprocess_cosmo_zarr.py in your working directory
micromamba run -n dl-torch python /work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/WeatherAdaptSR/data/preprocessing_COSMO.py