#!/bin/bash
#SBATCH -p debug
#SBATCH --nodes=1
#SBATCH -n 1
#SBATCH --mem=16384
#SBATCH --time=00:30:00
#SBATCH --job-name="test_memleak"
#SBATCH --mail-user=tobias.schmidt@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/scratch/tgschmidt/prep_data_%04a_stdout.txt
#SBATCH --error=/scratch/tgschmidt/prep_data_%04a_stderr.txt

#run the simple test
/home/tgschmidt/tf_gpu_env/bin/python -u /home/tgschmidt/hail/src/temp_categorizer.py

