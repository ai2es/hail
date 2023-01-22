#!/bin/bash

#SBATCH --partition=ai2es
#SBATCH --nodelist=c830
# Thread count:
#SBATCH --cpus-per-task=3
# memory in MB
#SBATCH --mem=25600
# The %04a is translated into a 4-digit number that encodes the SLURM_ARRAY_TASK_ID
#SBATCH --output=/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour-128_size-more_fields/slurm_output/out/prep_train_55_%04a_stdout.txt
#SBATCH --error=/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour-128_size-more_fields/slurm_output/err/prep_train_55_%04a_stderr.txt
#SBATCH --time=03:00:00
#SBATCH --job-name=train_55_preprocess
#SBATCH --mail-user=tobias.schmidt@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/tgschmidt/hail/src
#SBATCH --array=0-39%6
#
#################################################
# $SLURM_ARRAY_TASK_ID

/home/tgschmidt/sn_env/bin/python -u preprocessor.py -m -t -n --run_num $SLURM_ARRAY_TASK_ID --selected_time 11 --output_ds_dir /ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour-128_size-more_fields/patches/train/tf_datasets_55


