#!/bin/bash
#SBATCH -p debug
#SBATCH --nodes=1
#SBATCH -n 20
#SBATCH --mem-per-cpu=1000
#SBATCH --time=00:30:00
#SBATCH --job-name="ml_output"
#SBATCH --mail-user=lydiaks2@illinois.edu
#SBATCH --mail-type=ALL
#SBATCH --mail-type=END
#SBATCH --output=job_output/R-%x.%j.out
#SBATCH --error=job_output/R-%x.%j.err
#SBATCH --array=0-100%5

source .bashrc
bash
conda activate hagelslag

python generate_output.py $SLURM_ARRAY_TASK_ID