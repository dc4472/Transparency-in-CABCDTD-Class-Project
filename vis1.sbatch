#!/bin/bash
#SBATCH --requeue
#SBATCH --job-name=breast_cancer_model
#SBATCH --output=model_output.log
#SBATCH --error=model_error.log
#SBATCH --account=cs_ga_3033_102-2025sp
#SBATCH --partition=c12m85-a100-1       # Try this or stake_a100_2 / v100 if it fails
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00                      # Adjust time as needed
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dc4472@nyu.edu       # Replace with your email

# Load the appropriate Python and CUDA modules
module purge
module load python/3.9
module load cuda/11.7  # Adjust to what's supported on your cluster if needed

# Create a virtual environment if not exists
ENV_DIR="$HOME/Transparency-in-CABCDTD-Class-Project/pathology_env"
if [ ! -d "$ENV_DIR" ]; then
  python -m venv $ENV_DIR
  source $ENV_DIR/bin/activate
  pip install --upgrade pip
  pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
  pip install pandas numpy matplotlib seaborn opencv-python pydicom scikit-learn Pillow timm
else
  source $ENV_DIR/bin/activate
fi

# Go to the project directory
cd /home/dc4472/Transparency-in-CABCDTD-Class-Project/

# Run the script
python full_image_size_cropped_area.py
