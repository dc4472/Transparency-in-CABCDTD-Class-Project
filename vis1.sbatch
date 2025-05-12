#!/bin/bash
#SBATCH --job-name=run_image_processing
#SBATCH --output=output.log
#SBATCH --error=error.log
#SBATCH --account=cs_ga_3033_102-2025sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# Load modules
module purge
module load python/3.9

# Activate your environment from scratch
source /scratch/dc4472/pathology_env/bin/activate

# Move to project directory
cd ~/Transparency-in-CABCDTD-Class-Project

# Run your script
python full_image_size_cropped_area.py
