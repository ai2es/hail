#!/bin/bash

#SBATCH --partition=ai2es
# Thread count:
#SBATCH --cpus-per-task=1
# memory in MB
#SBATCH --mem=4096
# The %04a is translated into a 4-digit number that encodes the SLURM_ARRAY_TASK_ID
#SBATCH --output=/scratch/tgschmidt/tensorboard_stdout.txt
#SBATCH --error=/scratch/tgschmidt/tensorboard_stderr.txt
#SBATCH --time=02:00:00
#SBATCH --job-name=tensorboard
#SBATCH --mail-user=tobias.schmidt@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --array=0-0
#
#################################################
/home/tgschmidt/tf_gpu_env/bin/tensorboard --logdir="/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/trained_at_init_time/saved_models/unet_3plus/tensorboard_logdir" --port=6069 --bind_all


