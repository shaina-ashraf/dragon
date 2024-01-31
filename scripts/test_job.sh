#!/bin/bash
#SBATCH --job-name=sh_dragon
#SBATCH --partition=short
#SBATCH --time=00:02:00  # 2 minutes
#SBATCH --mem=4G         # 1 GB of RAM
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a100:1  # Request 1 a100 GPU

#SBATCH --output=/home/ashrafs/projects/dragon/scripts/logs/%x-%j.log
#SBATCH --error=/home/ashrafs/projects/dragon/scripts/logs/%x-%j.err

# Load Conda environment
source /home/ashrafs/miniconda3/etc/profile.d/conda.sh
conda activate dragon2

# Load CUDA module
module load cuda/11.1
# Run your test command
echo "Starting test run at: $(date)"
# Replace this with the command to run your script
# python my_script.py
bash run_train__fever.sh
echo "Test run completed at: $(date)"
