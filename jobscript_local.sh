#!/bin/bash -l
#SBATCH --account=torch_pr_292_courant
#SBATCH --cpus-per-task=4
<<<<<<< HEAD
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm_%A.out
#SBATCH --error=logs/slurm_%A.err
#SBATCH --mem=100G
=======
#SBATCH --time=2:00:00
#SBATCH --output=logs/slurm_%A.out
#SBATCH --error=logs/slurm_%A.err
#SBATCH --mem=60G
>>>>>>> refs/remotes/origin/main
#SBATCH --gres=gpu:l40s:1

# Pick the script for this array index

module load anaconda3/2025.06
source $(conda info --base)/etc/profile.d/conda.sh

conda activate desc
mkdir -p local
python -u optimization_local.py