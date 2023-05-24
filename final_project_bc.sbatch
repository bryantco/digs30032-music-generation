#!/bin/bash

#SBATCH --job-name=final_proj_bc
#SBATCH --output=final_proj_bc.out
#SBATCH --time=30:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=64G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=digs30032

module load python
conda activate digs
python3 final_project.py
