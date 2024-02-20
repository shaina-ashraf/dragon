#!/bin/bash

#SBATCH --job-name=fever_preprocess
#SBATCH --partition=owner_fb12
#SBATCH --mem=80G     
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/ashrafs/projects/dragon/scripts/logs/%x-%j.log
#SBATCH --error=/home/ashrafs/projects/dragon/scripts/logs/%x-%j.err

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=true
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

source /home/ashrafs/miniconda3/etc/profile.d/conda.sh
conda activate dragon2

module load cuda/11.1

echo "Starting pre-processing run at: $(date)"
python preprocess.py -p 8 --run fever
echo "Preprocessing run completed at: $(date)"
