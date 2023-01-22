#!/bin/bash

#SBATCH --partition=ai2es
#SBATCH --cpus-per-task=4
# memory in MB
#SBATCH --mem=32768
# The %04a is translated into a 4-digit number that encodes the SLURM_ARRAY_TASK_ID
#SBATCH --output=/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour-128_size-more_fields/slurm_output/out/predict_single_%04a_stdout.txt
#SBATCH --error=/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour-128_size-more_fields/slurm_output/err/predict_single_%04a_stderr.txt
#SBATCH --time=03:00:00
#SBATCH --job-name=predict_single
#SBATCH --mail-user=tobias.schmidt@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/tgschmidt/hail/src
#SBATCH --array=0-0
#
#################################################
# $SLURM_ARRAY_TASK_ID

/home/tgschmidt/sn_env/bin/python -u predictor.py -u -m --run_num $SLURM_ARRAY_TASK_ID


