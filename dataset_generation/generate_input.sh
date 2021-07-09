#!/bin/bash
#SBATCH -p normal
#SBATCH --nodes=1
#SBATCH -n 1
#SBATCH --mem-per-cpu=1000
#SBATCH --time=10:00:00
#SBATCH --job-name="ml_input"
#SBATCH --mail-user=lydiaks2@illinois.edu
#SBATCH --mail-type=ALL
#SBATCH --mail-type=END
#SBATCH --output=./output/R-%x.%j.out
#SBATCH --error=./errors/R-%x.%j.err
#SBATCH --array=0-100%5

source .bashrc
bash
conda activate hagelslag

python generate_input.py $SLURM_ARRAY_TASK_ID