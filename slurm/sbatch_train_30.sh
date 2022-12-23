#!/bin/bash

#SBATCH --partition=ai2es
#SBATCH --nodes=1
#SBATCH --nodelist=c829
# Thread count:
#SBATCH --cpus-per-task=12
# memory in MB
#SBATCH --mem=102400
# The %04a is translated into a 4-digit number that encodes the SLURM_ARRAY_TASK_ID
#SBATCH --output=/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour_fixed-128_size/slurm_output/out/train_model_30_%04a_stdout.txt
#SBATCH --error=/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour_fixed-128_size/slurm_output/err/train_model_30_%04a_stderr.txt
#SBATCH --time=72:00:00
#SBATCH --job-name=30_nowcasting
#SBATCH --mail-user=tobias.schmidt@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/tgschmidt/hail/src
#SBATCH --array=0-0
#
#################################################

/home/tgschmidt/tf_gpu_env/bin/python -u hparam_search.py


