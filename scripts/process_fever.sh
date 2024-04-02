#!/bin/bash

#SBATCH --job-name=fever_full_preprocess
#SBATCH --partition=owner_fb12
#SBATCH --mem=80G     
#SBATCH --gres=gpu:1

#SBATCH --output=/home/ashrafs/projects/dragon/scripts/logs/%x-%j.log
#SBATCH --error=/home/ashrafs/projects/dragon/scripts/logs/%x-%j.err

export CUDA_VISIBLE_DEVICES=0

source /home/ashrafs/miniconda3/etc/profile.d/conda.sh
conda activate dragon2

module load cuda/11.1

echo "Starting pre-processing run at: $(date)"
#python preprocess.py -p 30 --run fever
python preprocess.py -p 50 --run fever

echo "Preprocessing run completed at: $(date)"
