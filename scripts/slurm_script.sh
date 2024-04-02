#!/bin/bash

#SBATCH --job-name=fever_preprocess_testing
#SBATCH --partition=owner_fb12  # Adjusted to a more generic "standard" partition for testing
#SBATCH --mem=55G  # Ensure this aligns with your test requirements and availability on the cluster
#SBATCH --gres=gpu:1  # Request for a GPU, ensure the cluster supports this and adjust type if needed

#SBATCH --output=/home/ashrafs/projects/dragon/scripts/logs/%x-%j.log
#SBATCH --error=/home/ashrafs/projects/dragon/scripts/logs/%x-%j.err
#SBATCH --time=02:00:00  # Adding a time limit for the job. Adjust based on expected run time

export CUDA_VISIBLE_DEVICES=0
# Keeping parallelism and threading environment variables commented out, as they're set for single-threaded
# Uncomment and adjust if your test requires specific configurations

source /home/ashrafs/miniconda3/etc/profile.d/conda.sh
conda activate dragon2

module load cuda/11.1

echo "Starting pre-processing run at: $(date)"
# Adjust python command based on your needs. Removed '-p 30' for a simple test
python preprocess.py --run fever

echo "Preprocessing run completed at: $(date)"
