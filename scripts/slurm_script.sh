#!/bin/bash
#
#SBATCH --job-name=sa_dragon
#SBATCH --output=/home/ashrafs/logs/%x-%j.log
#SBATCH --error=/home/ashrafs/logs/%x-%j.err
#SBATCH --mail-user=ashrafs@staff.uni-marburg.de
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=owner_fb12
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
source /home/ashrafs/miniconda3/etc/profile.d/conda.sh
conda activate dragon2
module load cuda/11.1
#bash scripts/run_eval__csqa.sh
bash scripts/run_train__csqa.sh