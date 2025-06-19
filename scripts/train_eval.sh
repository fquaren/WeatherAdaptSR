#!/bin/bash -l
#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type NONE
#SBATCH --mail-user filippo.quarenghi@unil.ch

#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name UNet
#SBATCH --output outputs/%j
#SBATCH --error job_errors/%j

#SBATCH --partition gpu-gh
#SBATCH --gres gpu:1
#SBATCH --gres-flags enforce-binding
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 100G
#SBATCH --time 72:00:00

export SINGULARITY_BINDPATH="/work,/scratch,/users"

container_path="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/singularity/dl_gh200.sif"

# Train a single model

# model="UNet_MMD"
# singularity exec --nv "$container_path" \
#     python /work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/WeatherAdaptSR/cross-val-train.py \
#     --model "$model" \
#     --method "mmd"

# Train and evaluate different models in parallel

# models=("UNet" "UNet_DO" "UNet_BN" "UNet_DO_BN" "UNet_Noise")
# resume=("suyk" "m6dm" "mz3o" "o3zc" "rjdx")

models=("UNet_Noise_DO_BN" "UNet_Trainable_Noise" "UNet_MMD")
args=("" "" "--method mmd")

for i in "${!models[@]}"; do
    model="${models[$i]}"
    # exp="${resume[$i]}"
    extra_args=${args[$i]}
    (
        singularity exec --nv "$container_path" \
            python /work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/WeatherAdaptSR/cross-val-train.py \
            --model "$model" \
            $extra_args
            # --resume_exp "$exp"
        
        # singularity exec --nv "$container_path" \
        #     python /work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/WeatherAdaptSR/cross-evaluate.py \
        #     --device "cuda" \
        #     --model "$model" \
        #     --exp_path "/scratch/fquareng/experiments/cross-val-100/$exp"
    ) &
done
wait
