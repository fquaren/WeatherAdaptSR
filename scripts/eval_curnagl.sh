#!/bin/bash -l
#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type ALL
#SBATCH --mail-user filippo.quarenghi@unil.ch

#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name all+single
#SBATCH --output outputs/%j
#SBATCH --error job_errors/%j

#SBATCH --partition gpu
#SBATCH --gres gpu:2
#SBATCH --gres-flags enforce-binding
#SBATCH --nodes 1
#SBATCH --ntasks 2
#SBATCH --mem 100G
#SBATCH --time 5:00:00

module load singularityce/4.1.0
container_path="/users/fquareng/singularity/dl_curnagl.sif"
export SINGULARITY_BINDPATH="/work,/scratch,/users"

models=("UNet" "UNet")
methods=("single" "all")
exp_dir="/scratch/fquareng/experiments/single-all"
experiments=("$exp_dir/" "$exp_dir/" "$exp_dir/" "$exp_dir/")

for i in "${!models[@]}"; do
    model="${models[$i]}"
    method="${methods[$i]}"
    exp="${experiments[$i]}"
    singularity exec --nv "$container_path" \
        python /work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/WeatherAdaptSR/cross-evaluate.py \
        --device "cuda" \
        --model "$model" \
        --exp_path "$exp" \
        --method "$method"
done
wait