#!/bin/bash
#SBATCH --job-name=breast_cancer_model
#SBATCH --output=model_output.log
#SBATCH --error=model_error.log
#SBATCH --account=cs_ga_3033_102-2025sp
#SBATCH --partition=g2-standard-24           # Or any other GPU partition available to you
#SBATCH --gres=gpu:1                         # Request 1 GPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00                      # Adjust time as needed
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your.email@nyu.edu       # Replace with your email

# Load the appropriate Python and CUDA modules
module purge
module load python/3.9
module load cuda/11.7  # Adjust to what's supported on your cluster if needed

# Create a virtual environment if not exists
ENV_DIR="$HOME/Transparency-in-CABCDTD-Class-Project/breast_cancer_env"
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
python model_development_and_evaluation.py
