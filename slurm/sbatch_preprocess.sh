#!/bin/bash

#SBATCH --partition=ai2es
# Thread count:
#SBATCH --cpus-per-task=1
# memory in MB
#SBATCH --mem=65536
# The %04a is translated into a 4-digit number that encodes the SLURM_ARRAY_TASK_ID
#SBATCH --output=/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/trained_at_init_time_2022_08_09/slurm_output/out/prep_data_%04a_stdout.txt
#SBATCH --error=/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/trained_at_init_time_2022_08_09/slurm_output/err/prep_data_%04a_stderr.txt
#SBATCH --time=00:45:00
#SBATCH --job-name=nowcasting_preprocess
#SBATCH --mail-user=tobias.schmidt@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/tgschmidt/hail/src
#SBATCH --array=0-14
#
#################################################
# $SLURM_ARRAY_TASK_ID

/home/tgschmidt/sn_env/bin/python -u preprocessor.py --run_num $SLURM_ARRAY_TASK_ID


