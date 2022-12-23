#!/bin/bash

#SBATCH --partition=normal
# Thread count:
#SBATCH --cpus-per-task=1
# memory in MB
#SBATCH --mem=15360
# The %04a is translated into a 4-digit number that encodes the SLURM_ARRAY_TASK_ID
#SBATCH --output=/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour_fixed-128_size/slurm_output/out/prep_test_%04a_stdout.txt
#SBATCH --error=/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour_fixed-128_size/slurm_output/err/prep_test_%04a_stderr.txt
#SBATCH --time=03:00:00
#SBATCH --job-name=test_preprocess
#SBATCH --mail-user=tobias.schmidt@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/tgschmidt/hail/src
#SBATCH --array=0-26
#
#################################################
# $SLURM_ARRAY_TASK_ID

/home/tgschmidt/sn_env/bin/python -u preprocessor.py -m -t -s -d -p --run_num $SLURM_ARRAY_TASK_ID


