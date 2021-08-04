#!/bin/bash
#SBATCH -p debug
#SBATCH --nodes=1-1
#SBATCH -n 10
#SBATCH --mem-per-cpu=1000
#SBATCH --time=00:30:00
#SBATCH --job-name="runfile"
#SBATCH --mail-user=robjk3-22@rhodes.edu
#SBATCH --mail-type=ALL
#SBATCH --mail-type=END
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.err

source /home/jordot45/.bashrc
bash 
conda activate hagelslag
python mesh_output.py
