#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --job-name=hpc_test
#SBATCH --mail-user=dc4472@nyu.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=output_hpc_test.log
#SBATCH --error=error_hpc_test.log

# Load container environment
module purge

# Run the container with your overlay and Python script
singularity exec --nv \
  --overlay /scratch/$USER/python-singularity/my_python_env.ext3:ro \
  /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c "
    source /ext3/env.sh;
    conda activate myenv;
    echo '🧪 Which Python:'; which python;
    echo '🧪 Python version:'; python --version;
    python /home/$USER/Transparency-in-CABCDTD-Class-Project/HPCtest.py
  "
