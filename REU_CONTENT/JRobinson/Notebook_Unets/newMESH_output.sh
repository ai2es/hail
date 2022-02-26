#!/bin/bash
#SBATCH -p normal
#SBATCH --nodes=1
#SBATCH -n 20
#SBATCH --mem-per-cpu=900
#SBATCH --time=05:30:00
#SBATCH --job-name="runfile"
#SBATCH --mail-user=robjk3-22@rhodes.edu
#SBATCH --mail-type=ALL
#SBATCH --mail-type=END
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.err
#SBATCH --array=0-5%5

source /home/jordot45/.bashrc
bash 
conda activate hagelslag
python newMESH_output.py $SLURM_ARRAY_TASK_ID 