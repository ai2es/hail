#!/bin/bash
#SBATCH -p debug_5min
#SBATCH --nodes=1-1
#SBATCH -n 4
#SBATCH --mem-per-cpu=1000
#SBATCH --time=00:5:00
#SBATCH --job-name="unet_test"
#SBATCH --mail-user=robjk3_22@rhodes.edu
#SBATCH --mail-type=ALL
#SBATCH --mail-type=END
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.err

source /home/jordot45/.bashrc
bash 
conda activate hagelslag
python unet_regression_test7.py