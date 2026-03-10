#!/bin/bash -l
#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type ALL
#SBATCH --mail-user filippo.quarenghi@unil.ch
#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name eval_uda_unet
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

BASE_DIR="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/WeatherAdaptSR"
W1_PATH="${BASE_DIR}/covariate_shift_analysis/normalized/Wasserstein_1D_test.npy"

# 1. Define the identical explicit list of source and target couples
COUPLES=(
    "europe_west europe_west" # Baseline
    "europe_west blacksea" # Easy (Norm. W_1≈0.00)
    "europe_west horn-of-africa" # Medium (Norm. W_1≈0.05)
    "europe_west melanesia" # Hard (Norm. W_1≈0.16)
)

# 2. Define the identical adaptation methods
METHODS=(
    "none" 
    "coral" 
    "mmd"
    "sinkhorn"
    "spectral"
    "fourier"
)

ARCHITECTURE="unet"

echo "====================================================="
echo "Starting Targeted Evaluation Pipeline"
echo "Architecture: $ARCHITECTURE"
echo "====================================================="

echo "--- Evaluation Phase ---"
for COUPLE in "${COUPLES[@]}"; do
    read -r SOURCE_DOMAIN TARGET_DOMAIN <<< "$COUPLE"
    
    for ADAPTATION_METHOD in "${METHODS[@]}"; do
        echo "Evaluating $ARCHITECTURE ($ADAPTATION_METHOD) | $SOURCE_DOMAIN -> $TARGET_DOMAIN"
        
        # singularity exec --nv "$container_path" python ${BASE_DIR}/evaluate.py \
        #     --action evaluate \
        #     --source "$SOURCE_DOMAIN" \
        #     --target "$TARGET_DOMAIN" \
        #     --architecture "$ARCHITECTURE" \
        #     --adaptation_method "$ADAPTATION_METHOD"

        python ${BASE_DIR}/evaluate.py \
            --action evaluate \
            --source "$SOURCE_DOMAIN" \
            --target "$TARGET_DOMAIN" \
            --architecture "$ARCHITECTURE" \
            --adaptation_method "$ADAPTATION_METHOD"

    done
done

echo "--- Generalization Metrics Computation Phase ---"
for ADAPTATION_METHOD in "${METHODS[@]}"; do
    echo "Computing domain metrics for method: $ADAPTATION_METHOD"
    
    # singularity exec --nv "$container_path" python ${BASE_DIR}/evaluate.py \
    #     --action compute_metrics \
    #     --w1_path "$W1_PATH" \
    #     --architecture "$ARCHITECTURE" \
    #     --adaptation_method "$ADAPTATION_METHOD"

    python ${BASE_DIR}/evaluate.py \
        --action compute_metrics \
        --w1_path "$W1_PATH" \
        --architecture "$ARCHITECTURE" \
        --adaptation_method "$ADAPTATION_METHOD"
done

echo "Job completed successfully."