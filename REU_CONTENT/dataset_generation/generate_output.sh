#!/bin/bash
#SBATCH -p normal
#SBATCH --nodes=1
#SBATCH -n 5
#SBATCH --mem-per-cpu=1000
#SBATCH --time=05:00:00
#SBATCH --job-name="apr28"
#SBATCH --mail-user=lydiaks2@illinois.edu
#SBATCH --mail-type=ALL
#SBATCH --mail-type=END
#SBATCH --output=output/R-%x.%j.out
#SBATCH --error=errors/R-%x.%j.err
#SBATCH --array=5-9%5

source .bashrc
bash
conda activate hagelslag

python generate_output.py $SLURM_ARRAY_TASK_ID