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
#SBATCH --mem 500G
#SBATCH --time 72:00:00

export SINGULARITY_BINDPATH="/work,/scratch,/users"
container_path="/users/fquareng/singularity/dl_gh200.sif"

models=("HomoscedasticUNet_BN_Dropout")
methods=("single")
seeds=(0)

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# exp_dir="/scratch/fquareng/experiments/single-10x"
# experiments=("$exp_dir/8qd3" "$exp_dir/ahc0" "$exp_dir/epsf" "$exp_dir/h78o" "$exp_dir/jfk5" "$exp_dir/l778" "$exp_dir/oxjb" "$exp_dir/rnjb" "$exp_dir/rrqg" "$exp_dir/x586") # "$exp_dir/" "$exp_dir/" "$exp_dir/")
# exp="${experiments[$i]}"
# --resume_exp "$exp" \

for i in "${!seeds[@]}"; do
    model="${models[$i]}"
    method="${methods[$i]}"
    seed="${seeds[$i]}"
    
    singularity exec --nv "$container_path" \
        python /work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/WeatherAdaptSR/cross-val-train.py \
        --model "$model" \
        --method "$method" \
        --seed "$seed" &
done
wait
