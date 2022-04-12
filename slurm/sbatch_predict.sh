#!/bin/bash

#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
# memory in MB
#SBATCH --mem=8192
# The %04a is translated into a 4-digit number that encodes the SLURM_ARRAY_TASK_ID
#SBATCH --output=/home/tgschmidt/slurm_output/out/REU_predict_%04a_stdout.txt
#SBATCH --error=/home/tgschmidt/slurm_output/err/REU_predict_%04a_stderr.txt
#SBATCH --time=00:10:00
#SBATCH --job-name=predict
#SBATCH --mail-user=tobias.schmidt@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/tgschmidt/hail/src
#SBATCH --array=0-0
#
#################################################
# $SLURM_ARRAY_TASK_ID

/home/tgschmidt/tgs-env/bin/python -u modeler.py


