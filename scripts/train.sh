#!/bin/bash -l
#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type ALL
#SBATCH --mail-user filippo.quarenghi@unil.ch

#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name train
#SBATCH --output outputs/%j
#SBATCH --error job_errors/%j

#SBATCH --partition gpu
#SBATCH --gres gpu:1
#SBATCH --gres-flags enforce-binding
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 12
#SBATCH --mem 350G
#SBATCH --time 72:00:00

source /users/fquareng/.bashrc
micromamba activate dl-torch

# DOMAINS=(
#   "africa-south" "amazon-basin" "arabian-peninsula" "australasia-east"
#   "blacksea" "cape-horn" "caribbean" "east-asia-north-east"
#   "east-asia-south" "europe_west" "horn-of-africa" "melanesia"
#   "northamerica-east" "northamerica-west" "southamerica-east"
#   "southeastasia-west" "tibetan-plateau" "west-africa"
# )

DOMAINS=(
  #"amazon-basin" 
  #"arabian-peninsula" 
  #"australasia-east"
  #"blacksea" 
  #"cape-horn" 
  "caribbean" 
  #"east-asia-north-east"
  #"east-asia-south" 
  "europe_west" 
  "horn-of-africa" 
  #"melanesia"
  #"northamerica-east" 
  #"northamerica-west" 
  #"southamerica-east"
  #"southeastasia-west" 
  #"tibetan-plateau" 
  #"west-africa"
)

# Train all 18 models independently
for SOURCE in "${DOMAINS[@]}"; do
    echo "Training model on source domain: $SOURCE"
    python /work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/WeatherAdaptSR/train.py --source "$SOURCE" --subset_size 1000
done
