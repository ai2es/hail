#!/bin/bash

#SBATCH --partition=ai2es
#SBATCH --nodes=1
# Thread count:
#SBATCH --cpus-per-task=8
# memory in MB
#SBATCH --mem=81920
# The %04a is translated into a 4-digit number that encodes the SLURM_ARRAY_TASK_ID
#SBATCH --output=/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/no_refl_trained_at_init_time_2022_08_03/slurm_output/out/train_model_%04a_stdout.txt
#SBATCH --error=/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/no_refl_trained_at_init_time_2022_08_03/slurm_output/err/train_model_%04a_stderr.txt
#SBATCH --time=48:00:00
#SBATCH --job-name=nowcasting
#SBATCH --mail-user=tobias.schmidt@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/tgschmidt/hail/src
#SBATCH --array=0-0
#
#################################################
# $SLURM_ARRAY_TASK_ID
# Used to use --exclusive tag

/home/tgschmidt/sn_env/bin/python -u modeler.py -t


