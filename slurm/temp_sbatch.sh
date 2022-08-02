#!/bin/bash

#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
# memory in MB
#SBATCH --mem=4096
# The %04a is translated into a 4-digit number that encodes the SLURM_ARRAY_TASK_ID
#SBATCH --output=/home/tgschmidt/test_patch_run_%04a_stdout.txt
#SBATCH --error=/home/tgschmidt/test_patch_run_%04a_stderr.txt
#SBATCH --time=01:00:00
#SBATCH --job-name=fix_patches
#SBATCH --mail-user=tobias.schmidt@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/tgschmidt/hail/src
#SBATCH --array=0-0
#
#################################################
/home/tgschmidt/sn_env/bin/python -u temp_categorizer.py


