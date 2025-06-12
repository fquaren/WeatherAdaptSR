#!/bin/bash -l
#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type ALL
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

container_path="/users/fquareng/singularity/dl_gh200.sif"

models=("UNet" "UNet_DO" "UNet_BN" "UNet_DO_BN" "UNet_Noise" "UNet_Noise_DO_BN")

for model in "${models[@]}"; do
    (
        exp_path=$(singularity exec --nv "$container_path" python /work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/WeatherAdaptSR/cross-val-train.py --model "$model")
        
        # Optional: Add logging or checks
        echo "Model: $model, Experiment path: $exp_path"

        singularity exec --nv "$container_path" python /work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/WeatherAdaptSR/cross-evaluate.py --device "cuda" --exp_path "$exp_path"
    ) &
done

wait