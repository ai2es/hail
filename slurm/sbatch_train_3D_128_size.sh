#!/bin/bash

#SBATCH --partition=ai2es
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
# Thread count:
#SBATCH --cpus-per-task=16
# memory in MB
#SBATCH --mem=307200
# The %04a is translated into a 4-digit number that encodes the SLURM_ARRAY_TASK_ID
#SBATCH --output=/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/slurm_output/out/train_refl_nolightning_fold_0000_%04a_stdout.txt
#SBATCH --error=/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/slurm_output/err/train_refl_nolightning_fold_0000_%04a_stderr.txt
#SBATCH --time=12:00:00
#SBATCH --job-name=refl_nowcasting
#SBATCH --mail-user=tobias.schmidt@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/tgschmidt/hail/src
#SBATCH --array=0-0
#
#################################################
/home/tgschmidt/tf_gpu_env/bin/python hparam_search.py


