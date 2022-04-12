#!/bin/bash

#SBATCH --partition=ai2es_a100_2
#SBATCH --exclusive
#SBATCH --nodes=1
# Thread count:
#SBATCH --cpus-per-task=32
# memory in MB
#SBATCH --mem=40960
# The %04a is translated into a 4-digit number that encodes the SLURM_ARRAY_TASK_ID
#SBATCH --output=/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/REU_run/slurm_output/out/REU_MODEL_%04a_stdout.txt
#SBATCH --error=/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/REU_run/slurm_output/err/REU_MODEL_%04a_stderr.txt
#SBATCH --time=65:00:00
#SBATCH --job-name=nowcasting
#SBATCH --mail-user=tobias.schmidt@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/tgschmidt/hail/src
#SBATCH --array=0-0
#
#################################################
# $SLURM_ARRAY_TASK_ID

/home/tgschmidt/tgs-env/bin/python -u modeler.py


