#!/bin/bash -l
#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type ALL
#SBATCH --mail-user filippo.quarenghi@unil.ch

#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name plot_metrics
#SBATCH --output outputs/%j
#SBATCH --error job_errors/%j

#SBATCH --partition cpu
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 2
#SBATCH --mem 16G
#SBATCH --time 01:00:00

source /users/fquareng/.bashrc
micromamba activate dl-torch

BASE_DIR="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/WeatherAdaptSR"
ARCHITECTURE="unet"

# Define the adaptation methods to plot
METHODS=(
    "none" 
    "coral" 
    "mmd"
    "sinkhorn"
    "spectral"
    "fourier"
)

echo "====================================================="
echo "Generating Evaluation Plots for $ARCHITECTURE"
echo "====================================================="

for ADAPTATION_METHOD in "${METHODS[@]}"; do
    echo "--- Processing method: $ADAPTATION_METHOD ---"
    
    python ${BASE_DIR}/plot.py \
        --architecture "$ARCHITECTURE" \
        --adaptation_method "$ADAPTATION_METHOD"
        
done

echo "Plotting complete."