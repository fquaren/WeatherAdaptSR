#!/bin/bash -l
#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type ALL
#SBATCH --mail-user filippo.quarenghi@unil.ch
#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name train_uda
#SBATCH --output outputs/%j
#SBATCH --error job_errors/%j
#SBATCH --partition gpu-gh
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 48
#SBATCH --mem 150G
#SBATCH --time 72:00:00

export SINGULARITY_BINDPATH="/work,/scratch,/users"
export SINGULARITYENV_LD_PRELOAD="/opt/hpcx/ucc/lib/libucc.so.1:/opt/hpcx/ucx/lib/libucp.so.0:/opt/hpcx/ucx/lib/libucs.so.0" 
container_path="/users/fquareng/singularity/dl_gh200.sif"

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

# 1. Fetch variables from command line arguments
SOURCE_DOMAIN=$1
TARGET_DOMAIN=$2
ARCHITECTURE=${3:-unet} # Defaults to unet if a third argument is not provided
ADAPTATION_METHOD="" # Placeholder for future use, currently not used in the training script
SUBSET_SIZE=10000

if [ -z "$SOURCE_DOMAIN" ] || [ -z "$TARGET_DOMAIN" ]; then
    echo "Error: Source and Target domains must be provided."
    echo "Usage: sbatch run_train.sh <source_domain> <target_domain> [architecture]"
    exit 1
fi

echo "====================================================="
echo "Training $ARCHITECTURE model"
echo "Source: $SOURCE_DOMAIN -> Target: $TARGET_DOMAIN"
echo "====================================================="

singularity exec --nv "$container_path" python /work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/WeatherAdaptSR/train_gh200.py \
    --source "$SOURCE_DOMAIN" \
    --target "$TARGET_DOMAIN" \
    --architecture "$ARCHITECTURE" \
    --adaptation_method "$ADAPTATION_METHOD"
    --subset_size "$SUBSET_SIZE"