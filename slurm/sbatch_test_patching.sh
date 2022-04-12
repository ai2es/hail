#!/bin/bash

#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
# memory in MB
#SBATCH --mem=8192
# The %04a is translated into a 4-digit number that encodes the SLURM_ARRAY_TASK_ID
#SBATCH --output=/home/tgschmidt/slurm_output/out/REU_test_patch_run_%04a_stdout.txt
#SBATCH --error=/home/tgschmidt/slurm_output/err/REU_test_patch_run_%04a_stderr.txt
#SBATCH --time=02:30:00
#SBATCH --job-name=generate_patches
#SBATCH --mail-user=tobias.schmidt@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/tgschmidt/hail/src
#SBATCH --array=0-24
#
#################################################
/home/tgschmidt/tgs-env/bin/python patcher.py --config_path /home/tgschmidt/hail/configs/patcher_oscer_testing.cfg --run_num $SLURM_ARRAY_TASK_ID

