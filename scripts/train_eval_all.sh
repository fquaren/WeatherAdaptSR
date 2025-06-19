#!/bin/bash -l
#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type NONE
#SBATCH --mail-user filippo.quarenghi@unil.ch

#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name UNet
#SBATCH --output outputs/%j
#SBATCH --error job_errors/%j

#SBATCH --partition gpu
#SBATCH --gres gpu:2
#SBATCH --gres-flags enforce-binding
#SBATCH --nodes 2
#SBATCH --ntasks 2
#SBATCH --ntasks-per-node 1
#SBATCH --mem 100G
#SBATCH --time 72:00:00

export SINGULARITY_BINDPATH="/work,/scratch,/users"

container_path="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/singularity/dl_gh200.sif"

# Train a single model
model="UNet"
singularity exec --nv "$container_path" \
    python /work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/WeatherAdaptSR/train_all.py \
    --model "$model" \

# singularity exec --nv "$container_path" \
#     python /work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/WeatherAdaptSR/evaluate_all.py \
#     --device "cuda" \
#     --model "$model" \
#     --exp_path "/scratch/fquareng/experiments/all/$exp"