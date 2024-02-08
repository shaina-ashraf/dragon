#!/bin/bash
#SBATCH --job-name=fev_preprocess
#SBATCH --partition=short
#SBATCH --time=1:00:00  
#SBATCH --mem=32G         # 1 GB of RAM
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a100:1  # Request 1 a100 GPU

#SBATCH --output=/home/ashrafs/projects/dragon/scripts/logs/%x-%j.log
#SBATCH --error=/home/ashrafs/projects/dragon/scripts/logs/%x-%j.err

# Load Conda environment
source /home/ashrafs/miniconda3/etc/profile.d/conda.sh
conda activate dragon2

# Load CUDA module
module load cuda/11.1

#cd ..  # Go up one directory to where dragon.py is located

# Run your test command
echo "Starting test run at: $(date)"
# Replace this with the command to run your script
python preprocess.py --run fever

#cd -  # Return to the original directory (optional)

echo "Test run completed at: $(date)"
