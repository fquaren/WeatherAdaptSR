#!/bin/bash -l
#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type ALL
#SBATCH --mail-user filippo.quarenghi@unil.ch

#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name UNet-opt
#SBATCH --output outputs/%j
#SBATCH --error job_errors/%j

#SBATCH --partition gpu
#SBATCH --gres gpu:2
#SBATCH --gres-flags enforce-binding
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 50G
#SBATCH --time 48:00:00


module load singularityce/4.1.0
export SINGULARITY_BINDPATH="/scratch,/dcsrsoft,/users,/work,/reference"
singularity run --nv /dcsrsoft/singularity/containers/pytorch/pytorch-ngc-24.05-2.4.sif

source /users/fquareng/.bashrc
micromamba activate dl

# Run the training script
micromamba run -n dl python /work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/WeatherAdaptSR/cross-val-train.py
# Read experiement path from experiment.csv file and run cross-evaluation
exp_path=$(tail -n 1 /work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/WeatherAdaptSR/experiment.csv | awk -F, '{print $NF}')
micromamba run -n dl python /work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/WeatherAdaptSR/cross-evaluate.py --device "cuda" --exp_path "$exp_path"
