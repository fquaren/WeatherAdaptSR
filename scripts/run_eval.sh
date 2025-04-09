#!/bin/bash -l
#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type ALL
#SBATCH --mail-user filippo.quarenghi@unil.ch

#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name eval_12z4
#SBATCH --output outputs/%j
#SBATCH --error job_errors/%j

#SBATCH --partition gpu
#SBATCH --gpus 1
#SBATCH --gres-flags enforce-binding
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 100G
#SBATCH --time 24:00:00


module load singularityce/4.1.0
export SINGULARITY_BINDPATH="/scratch,/dcsrsoft,/users,/work,/reference"
singularity run --nv /dcsrsoft/singularity/containers/pytorch/pytorch-ngc-24.05-2.4.sif

source /users/fquareng/.bashrc
micromamba activate dl

micromamba run -n dl python /work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/WeatherAdaptSR/cross-evaluate.py --device "cuda" --exp_path "/scratch/fquareng/experiments/UNet_experiments_12/12z4"

# Evaluate all experiments in the folder
# base_path=/scratch/fquareng/experiments/UNet_experiments_12
# folders=$(ls -d $base_path/*/)
# echo "Folders to be evaluated:"
# echo "$folders"
# # Loop through each folder and run the evaluation script
# for folder in $folders; do
#     echo "Evaluating experiment in folder: $folder"
#     # Run the evaluation script
#     micromamba run -n dl python /work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/WeatherAdaptSR/cross-evaluate.py --device "cuda" --exp_path $folder
#     # Check if the evaluation script ran successfully
#     if [ $? -eq 0 ]; then
#         echo "Evaluation completed successfully for folder: $folder"
#     else
#         echo "Evaluation failed for folder: $folder"
#     fi
#     # Sleep command to avoid overloading the system
#     sleep 1
# done