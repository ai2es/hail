#!/bin/bash
#SBATCH -p normal
#SBATCH --nodes=1
#SBATCH -n 20
#SBATCH --mem-per-cpu=1000
#SBATCH --time=05:30:00
#SBATCH --job-name="ml_output"
#SBATCH --mail-user=lydiaks2@illinois.edu
#SBATCH --mail-type=ALL
#SBATCH --mail-type=END
#SBATCH --output=output/R-%x.%j.out
#SBATCH --error=errors/%x.%j.err
#SBATCH --array=0-100%5

source .bashrc
bash
conda activate hagelslag

python generate_output.py $SLURM_ARRAY_TASK_ID