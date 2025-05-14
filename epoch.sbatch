#!/bin/bash
#SBATCH --job-name=plot_accuracy
#SBATCH --output=log_plot_accuracy.out
#SBATCH --error=log_plot_accuracy.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dc4472@nyu.edu

# Load Singularity module
module purge


# Run container with overlay and conda environment
singularity exec --nv \
  --overlay /scratch/$USER/python-singularity/my_python_env.ext3:ro \
  /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c "
    source /ext3/env.sh;
    conda activate myenv;
    echo 'Job Started on $(hostname) at $(date)';
    echo 'Python path:'; which python;
    echo 'Python version:'; python --version;
    echo 'Running training accuracy plotting script...';
    python /home/$USER/Transparency-in-CABCDTD-Class-Project/epochPlots.py;
    echo 'Plotting complete';
    echo 'Job Finished at $(date)';
  "
