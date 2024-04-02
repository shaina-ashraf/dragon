#!/bin/bash

#SBATCH --job-name=fever_Roberta
#SBATCH --partition=owner_fb12
##SBATCH --partition=short
#SBATCH --mem=40GB     
#SBATCH --gres=gpu:1

#SBATCH --output=/home/ashrafs/projects/dragon/scripts/logs/%x-%j.log
#SBATCH --error=/home/ashrafs/projects/dragon/scripts/logs/%x-%j.err

export CUDA_VISIBLE_DEVICES=1

source /home/ashrafs/miniconda3/etc/profile.d/conda.sh
conda activate robert2-env

module load cuda/11.1
echo "Starting Roberta_experiments run at: $(date)"
python ../process_fever/Roberta_Model.py
#python ../check.py
echo "Roberta run completed at: $(date)"
