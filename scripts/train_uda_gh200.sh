#!/bin/bash -l
#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type ALL
#SBATCH --mail-user filippo.quarenghi@unil.ch
#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name train_uda_unet
#SBATCH --output outputs/%j
#SBATCH --error job_errors/%j
#SBATCH --partition gpu
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 12
#SBATCH --mem 150G
#SBATCH --time 72:00:00

# export SINGULARITY_BINDPATH="/work,/scratch,/users"
# export SINGULARITYENV_LD_PRELOAD="/opt/hpcx/ucc/lib/libucc.so.1:/opt/hpcx/ucx/lib/libucp.so.0:/opt/hpcx/ucx/lib/libucs.so.0" 
# container_path="/users/fquareng/singularity/dl_gh200.sif"

source /users/fquareng/.bashrc
micromamba activate dl-torch

COUPLES=(
    "europe_west europe_west" # Baseline
    "europe_west blacksea" # Easy (Norm. W_1≈0.00)
    "europe_west horn-of-africa" # Medium (Norm. W_1≈0.05)
    "europe_west melanesia" # Hard (Norm. W_1≈0.16)
)

# Added the new methods replacing adversarial learning
METHODS=(
    "none" 
    "coral" 
    "mmd"
    "sinkhorn"
    "spectral"
    "fourier"
)

ARCHITECTURE="unet"
SUBSET_SIZE=1000

for COUPLE in "${COUPLES[@]}"; do
    read -r SOURCE_DOMAIN TARGET_DOMAIN <<< "$COUPLE"
    
    for ADAPTATION_METHOD in "${METHODS[@]}"; do
        echo "====================================================="
        echo "Training $ARCHITECTURE model"
        echo "Source: $SOURCE_DOMAIN -> Target: $TARGET_DOMAIN"
        echo "Method: $ADAPTATION_METHOD"
        echo "====================================================="

        # singularity exec --nv "$container_path" python /work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/WeatherAdaptSR/train_gh200.py \
        #     --source "$SOURCE_DOMAIN" \
        #     --target "$TARGET_DOMAIN" \
        #     --architecture "$ARCHITECTURE" \
        #     --adaptation_method "$ADAPTATION_METHOD" \
        #     --subset_size "$SUBSET_SIZE"

        python /work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/WeatherAdaptSR/train_gh200.py \
            --source "$SOURCE_DOMAIN" \
            --target "$TARGET_DOMAIN" \
            --architecture "$ARCHITECTURE" \
            --adaptation_method "$ADAPTATION_METHOD" \
            --subset_size "$SUBSET_SIZE"
            
    done
done