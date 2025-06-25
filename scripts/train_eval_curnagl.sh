#!/bin/bash -l
#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type ALL
#SBATCH --mail-user filippo.quarenghi@unil.ch

#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name MMD
#SBATCH --output outputs/%j
#SBATCH --error job_errors/%j

#SBATCH --partition gpu
#SBATCH --gres gpu:2
#SBATCH --gres-flags enforce-binding
#SBATCH --nodes 2
#SBATCH --ntasks 2
#SBATCH --mem 100G
#SBATCH --time 72:00:00

module load singularityce/4.1.0  # Comment out if on GH200
container_path="/users/fquareng/singularity/dl_curnagl.sif"
export SINGULARITY_BINDPATH="/work,/scratch,/users"

## Train a single model
model="UNet_MMD"
method="mmd"
# resume="/scratch/fquareng/experiments/cross-val-v8/97cr"
exp=$(singularity exec --nv "$container_path" \
    python /work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/WeatherAdaptSR/cross-val-train.py \
    --model "$model" \
    --method "$method" \
)
singularity exec --nv "$container_path" \
    python /work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/WeatherAdaptSR/cross-evaluate.py \
    --device "cuda" \
    --model "$model" \
    --exp_path "$exp" \
    --method "$method" \
