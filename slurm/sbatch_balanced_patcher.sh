#!/bin/bash

#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
# memory in MB
#SBATCH --mem=4096
# The %04a is translated into a 4-digit number that encodes the SLURM_ARRAY_TASK_ID
#SBATCH --output=/home/tgschmidt/slurm_output/out/REU_run_%04a_stdout.txt
#SBATCH --error=/home/tgschmidt/slurm_output/err/REU_run_%04a_stderr.txt
#SBATCH --time=00:05:00
#SBATCH --job-name=generate_patches
#SBATCH --mail-user=tobias.schmidt@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/tgschmidt/hail/src
#SBATCH --array=0-0
#
#################################################
/home/tgschmidt/tgs-env/bin/python balanced_patcher.py --config_path /home/tgschmidt/hail/configs/balanced_patcher_oscer.cfg --run_num $SLURM_ARRAY_TASK_ID


