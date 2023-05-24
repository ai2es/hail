#!/bin/bash

#SBATCH --partition=ai2es
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
# Thread count:
#SBATCH --cpus-per-task=32
# memory in MB
#SBATCH --mem=409600
# The %04a is translated into a 4-digit number that encodes the SLURM_ARRAY_TASK_ID
#SBATCH --output=/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/slurm_output/out/train_3D_nogaus_fold_0000_%04a_stdout.txt
#SBATCH --error=/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/slurm_output/err/train_3D_nogaus_fold_0000_%04a_stderr.txt
#SBATCH --time=infinite
#SBATCH --job-name=nogaus_3D_nowcasting
#SBATCH --mail-user=tobias.schmidt@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/tgschmidt/hail/src
#SBATCH --array=0-0
#
#################################################
# $SLURM_ARRAY_TASK_IDs
# Used to use --exclusive and --nodelist=c732

/home/tgschmidt/tf_gpu_env/bin/python hparam_search.py


