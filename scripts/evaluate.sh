#!/bin/bash -l
#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type ALL
#SBATCH --mail-user filippo.quarenghi@unil.ch

#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name rainshift_generalization
#SBATCH --output outputs/%j
#SBATCH --error job_errors/%j

#SBATCH --partition gpu
#SBATCH --gres gpu:1
#SBATCH --gres-flags enforce-binding
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 4
#SBATCH --mem 350G
#SBATCH --time 72:00:00

source /users/fquareng/.bashrc
micromamba activate dl-torch

# Define project base path to avoid repetitive absolute paths
BASE_DIR="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/WeatherAdaptSR"

DOMAINS=(
  "africa-south" "amazon-basin" "arabian-peninsula" "australasia-east"
  "blacksea" "cape-horn" "caribbean" "east-asia-north-east"
  "east-asia-south" "europe_west" "horn-of-africa" "melanesia"
  "northamerica-east" "northamerica-west" "southamerica-east"
  "southeastasia-west" "tibetan-plateau" "west-africa"
)

echo "--- Evaluation ---"
for SOURCE in "${DOMAINS[@]}"; do
    for TARGET in "${DOMAINS[@]}"; do
        echo "Evaluating model trained on $SOURCE against target $TARGET"
        python ${BASE_DIR}/evaluate.py --action evaluate --source "$SOURCE" --target "$TARGET"
    done
done

echo "--- Phase 3: Generalization Metrics Computation ---"
# Adjust the w1_path to exactly where your pre-computed numpy array is saved
W1_PATH="${BASE_DIR}/covariate_shift_analysis/normalized/Wasserstein_1D_test.npy"
python ${BASE_DIR}/evaluate.py --action compute_metrics --w1_path "$W1_PATH"

echo "Job completed successfully."