#!/bin/bash

#SBATCH --partition=ai2es
# Thread count:
#SBATCH --cpus-per-task=1
# memory in MB
#SBATCH --mem=4096
#SBATCH --nodelist=c314
# The %04a is translated into a 4-digit number that encodes the SLURM_ARRAY_TASK_ID
#SBATCH --output=/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour_fixed/slurm_output/out/tensorboard_stdout.txt
#SBATCH --error=/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour_fixed/slurm_output/err/tensorboard_stderr.txt
#SBATCH --time=01:00:00
#SBATCH --job-name=tensorboard
#SBATCH --mail-user=tobias.schmidt@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --array=0-0
#
#################################################
/home/tgschmidt/tf_gpu_env/bin/tensorboard --logdir="/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour-more_fields-1_inch/saved_models/tensorboard_logdir" --port=6065 --bind_all


