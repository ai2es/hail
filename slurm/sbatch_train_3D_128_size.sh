#!/bin/bash

#SBATCH --partition=ai2es
#SBATCH --nodes=1
#SBATCH --nodelist=c829
#SBATCH --exclusive
# Thread count:
#SBATCH --cpus-per-task=64
# memory in MB
#SBATCH --mem=491520
# The %04a is translated into a 4-digit number that encodes the SLURM_ARRAY_TASK_ID
#SBATCH --output=/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour-more_fields-1_inch-balanced/slurm_output/out/train_model_loss_3plus_baserate_relu_%04a_stdout.txt
#SBATCH --error=/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour-more_fields-1_inch-balanced/slurm_output/err/train_model_loss_3plus_baserate_relu_%04a_stderr.txt
#SBATCH --time=60:00:00
#SBATCH --job-name=3D_nowcasting
#SBATCH --mail-user=tobias.schmidt@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/tgschmidt/hail/src
#SBATCH --array=0-0
#
#################################################
# $SLURM_ARRAY_TASK_ID
# Used to use --exclusive and --nodelist=c732

/home/tgschmidt/tf_gpu_env/bin/python -u hparam_search.py


