#!/bin/bash
#SBATCH --job-name=compare_models
#SBATCH --output=log_compare.out
#SBATCH --error=log_compare.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dc4472@nyu.edu

# Load environment modules
module purge

# Run Singularity container with overlay and execute comparison.py
singularity exec --nv \
  --overlay /scratch/$USER/python-singularity/my_python_env.ext3:ro \
  /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c "
    source /ext3/env.sh;
    conda activate myenv;
    echo '🧪 Job started on:' \$(hostname);
    echo '🔁 Running comparison.py';
    python /home/$USER/Transparency-in-CABCDTD-Class-Project/comparison.py;
    echo '✅ Done.';
  "
