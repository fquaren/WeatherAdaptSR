#!/bin/bash -l
#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type ALL
#SBATCH --mail-user filippo.quarenghi@unil.ch

#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name eval_uda
#SBATCH --output outputs/%j.out
#SBATCH --error job_errors/%j.err

#SBATCH --partition gpu
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 12
#SBATCH --mem 150G
#SBATCH --time 72:00:00

export SINGULARITY_BINDPATH="/work,/scratch,/users"
export SINGULARITYENV_LD_PRELOAD="/opt/hpcx/ucc/lib/libucc.so.1:/opt/hpcx/ucx/lib/libucp.so.0:/opt/hpcx/ucx/lib/libucs.so.0" 
container_path="/users/fquareng/singularity/dl_gh200.sif"

BASE_DIR="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/WeatherAdaptSR"

DOMAINS=(
  "amazon-basin" 
  "arabian-peninsula" 
  "australasia-east"
  "blacksea" 
  "cape-horn" 
  "caribbean" 
  "east-asia-north-east"
  "east-asia-south" 
  "europe_west" 
  "horn-of-africa" 
  "melanesia"
  "northamerica-east" 
  "northamerica-west" 
  "southamerica-east"
  "southeastasia-west" 
  "tibetan-plateau" 
  "west-africa"
)

# Fetch variables from command line arguments (with defaults)
ARCHITECTURE=${1:-unet}
ADAPTATION_METHOD=${2:-none}

echo "====================================================="
echo "Starting Evaluation Matrix"
echo "Architecture: $ARCHITECTURE"
echo "Adaptation Method: $ADAPTATION_METHOD"
echo "====================================================="

echo "--- Evaluation ---"
for SOURCE in "${DOMAINS[@]}"; do
    for TARGET in "${DOMAINS[@]}"; do
        echo "Evaluating $ARCHITECTURE ($ADAPTATION_METHOD) trained on $SOURCE against target $TARGET"
        singularity exec --nv "$container_path" python ${BASE_DIR}/evaluate.py \
            --action evaluate \
            --source "$SOURCE" \
            --target "$TARGET" \
            --architecture "$ARCHITECTURE" \
            --adaptation_method "$ADAPTATION_METHOD"
    done
done

echo "--- Generalization Metrics Computation ---"
# Adjust the w1_path to exactly where your pre-computed numpy array is saved
W1_PATH="${BASE_DIR}/covariate_shift_analysis/normalized/Wasserstein_1D_test.npy"

singularity exec --nv "$container_path" python ${BASE_DIR}/evaluate.py \
    --action compute_metrics \
    --w1_path "$W1_PATH" \
    --architecture "$ARCHITECTURE" \
    --adaptation_method "$ADAPTATION_METHOD"

echo "Job completed successfully."