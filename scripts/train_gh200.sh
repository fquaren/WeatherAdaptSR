#!/bin/bash -l
#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type ALL
#SBATCH --mail-user filippo.quarenghi@unil.ch

#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name UNet_mix
#SBATCH --output outputs/%j
#SBATCH --error job_errors/%j

#SBATCH --partition gpu-gh
#SBATCH --gres gpu:1
#SBATCH --gres-flags enforce-binding
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 250G
#SBATCH --time 72:00:00

export SINGULARITY_BINDPATH="/work,/scratch,/users"
container_path="/users/fquareng/singularity/dl_gh200.sif"

models=("UNet") #"UNet_DO_BN" "UNet_Noise" "UNet_MMD")
methods=("single") # "cross-val" "cross-val" "mmd")
seeds=(0 1 2 3 4 5 6 7 8 9)
# exp_dir="/scratch/fquareng/experiments/single-10x"
# experiments=("$exp_dir/8qd3" "$exp_dir/ahc0" "$exp_dir/epsf" "$exp_dir/h78o" "$exp_dir/jfk5" "$exp_dir/l778" "$exp_dir/oxjb" "$exp_dir/rnjb" "$exp_dir/rrqg" "$exp_dir/x586") # "$exp_dir/" "$exp_dir/" "$exp_dir/")
# exp="${experiments[$i]}"
# --resume_exp "$exp" \

for i in "${!seeds[@]}"; do
    model="UNet"
    method="single"
    seed="${seeds[$i]}"
    
    singularity exec --nv "$container_path" \
        python /work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/WeatherAdaptSR/cross-val-train.py \
        --model "$model" \
        --method "$method" \
        --seed "$seed" &
done
wait

# for i in "${!models[@]}"; do
#     model="${models[$i]}"
#     method="${methods[$i]}"
    
#     singularity exec --nv "$container_path" \
#         python /work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/WeatherAdaptSR/cross-val-train.py \
#         --model "$model" \
#         --method "$method" \

# done
# wait

# ## Train a single model
# model="UNet_MMD"
# method="mmd"
# exp=$(singularity exec --nv "$container_path" \
#     python /work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/WeatherAdaptSR/cross-val-train.py \
#     --model "$model" \
#     --method "$method" \
# )
# singularity exec --nv "$container_path" \
#     python /work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/WeatherAdaptSR/cross-evaluate.py \
#     --device "cuda" \
#     --model "$model" \
#     --exp_path "$exp" \
#     --method "$method" \

## Train and evaluate different models in parallel

# models=("UNet" "UNet_DO" "UNet_BN" "UNet_DO_BN" "UNet_Noise")
# resume=("suyk" "m6dm" "mz3o" "o3zc" "rjdx")
    # exp="${resume[$i]}"
    # --resume_exp "$exp"
# models=("UNet_Noise_DO_BN" "UNet_MMD")
# resume=("0s5v" "vkwn")
# args=("" "--method mmd")