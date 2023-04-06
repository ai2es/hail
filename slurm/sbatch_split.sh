#!/bin/bash

#SBATCH --partition=ai2es
# Thread count:
#SBATCH --cpus-per-task=8
# memory in MB
#SBATCH --mem=409600
# The %04a is translated into a 4-digit number that encodes the SLURM_ARRAY_TASK_ID
#SBATCH --output=/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-1_hour-1_inch-train_val_test-cross_val/slurm_output/out/split_%04a_stdout.txt
#SBATCH --error=/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-1_hour-1_inch-train_val_test-cross_val/slurm_output/err/split_%04a_stderr.txt
#SBATCH --time=10:00:00
#SBATCH --job-name=split
#SBATCH --mail-user=tobias.schmidt@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/tgschmidt/hail/src
#SBATCH --array=0-0
#
#################################################
num_name=$(printf "%04d" $SLURM_ARRAY_TASK_ID)

/home/tgschmidt/sn_env/bin/python -u splitter.py