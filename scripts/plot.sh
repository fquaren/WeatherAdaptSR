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
INPUT_CSV="${BASE_DIR}/results/detailed_generalization_metrics.csv"
OUTPUT_DIR="${BASE_DIR}/plots"

echo "--- Generating Evaluation Plots ---"
python ${BASE_DIR}/plot_generalization.py --input_csv "$INPUT_CSV" --output_dir "$OUTPUT_DIR"

echo "Plotting complete."