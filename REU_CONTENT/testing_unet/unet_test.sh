#!/bin/bash
#SBATCH -p debug_5min
#SBATCH --nodes=1-1
#SBATCH -n 4
#SBATCH --mem-per-cpu=1000
#SBATCH --time=00:05:00
#SBATCH --job-name="unet_test"
#SBATCH --mail-user=lydiaks2@illinois.edu
#SBATCH --mail-type=ALL

source ~/.bashrc
bash
conda activate hagelslag

python unet_test.py