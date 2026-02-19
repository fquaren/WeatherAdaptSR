#!/bin/bash -l
#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type ALL
#SBATCH --mail-user filippo.quarenghi@unil.ch

#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name trainmode
#SBATCH --output outputs/%j
#SBATCH --error job_errors/%j

#SBATCH --partition gpu-gh
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 48
#SBATCH --mem 400G
#SBATCH --time 72:00:00
#SBATCH --array=0-17

export SINGULARITY_BINDPATH="/work,/scratch,/users"
export SINGULARITYENV_LD_PRELOAD="/opt/hpcx/ucc/lib/libucc.so.1:/opt/hpcx/ucx/lib/libucp.so.0:/opt/hpcx/ucx/lib/libucs.so.0" 
container_path="/users/fquareng/singularity/dl_gh200.sif"

DOMAINS=(
  "africa-south" "amazon-basin" "arabian-peninsula" "australasia-east"
  "blacksea" "cape-horn" "caribbean" "east-asia-north-east"
  "east-asia-south" "europe_west" "horn-of-africa" "melanesia"
  "northamerica-east" "northamerica-west" "southamerica-east"
  "southeastasia-west" "tibetan-plateau" "west-africa"
)

# Extract the specific domain for this parallel task
SOURCE_DOMAIN=${DOMAINS[$SLURM_ARRAY_TASK_ID]}

echo "Training model on source domain: $SOURCE_DOMAIN"
singularity exec --nv "$container_path" python /work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/WeatherAdaptSR/train_gh200.py --source "$SOURCE_DOMAIN"
