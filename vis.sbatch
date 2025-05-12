#!/bin/bash
#SBATCH --requeue
#SBATCH --job-name=breast_cancer_pipeline
#SBATCH --output=pipeline_output.log
#SBATCH --error=pipeline_error.log
#SBATCH --account=cs_ga_3033_102-2025sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dc4472@nyu.edu

# Load Python + CUDA
module purge
module load python/3.9
module load cuda/11.7

# Activate Python environment
ENV_DIR="$HOME/Transparency-in-CABCDTD-Class-Project/breast_cancer_env"
source $ENV_DIR/bin/activate

# Move to project directory
cd /home/dc4472/Transparency-in-CABCDTD-Class-Project/

# 3. Run final image stats script
python full_image_size_cropped_area.py

# 2. Run R visualization (requires tidyverse, ggplot2, etc.)
module load R/4.2.2
Rscript R_visualization.R


