#!/bin/bash

#SBATCH --partition=normal
# Thread count:
#SBATCH --cpus-per-task=2
# memory in MB
#SBATCH --mem=32768
# The %04a is translated into a 4-digit number that encodes the SLURM_ARRAY_TASK_ID
#SBATCH --output=/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour-more_fields-1_inch-cross_val/slurm_output/out/prep_hailcast_%04a_stdout.txt
#SBATCH --error=/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour-more_fields-1_inch-cross_val/slurm_output/err/prep_hailcast_%04a_stderr.txt
#SBATCH --time=03:00:00
#SBATCH --job-name=hail_preprocess
#SBATCH --mail-user=tobias.schmidt@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/tgschmidt/hail/src
#SBATCH --array=0-26%6
#
#################################################
# $SLURM_ARRAY_TASK_ID

/home/tgschmidt/sn_env/bin/python -u preprocessor.py -s -d -t -p -u --run_num $SLURM_ARRAY_TASK_ID --examples "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour-more_fields-1_inch-cross_val/patches/test/unprocessed/examples/*" --labels "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour-more_fields-1_inch-cross_val/patches/test/unprocessed/labels/*" --output_ds_dir "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour-more_fields-1_inch-cross_val/patches/test/hailcast" --n_parallel_runs 27


