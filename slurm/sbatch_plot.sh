#!/bin/bash

#SBATCH --partition=ai2es
# Thread count:
#SBATCH --cpus-per-task=8
# memory in MB
#SBATCH --mem=153600
# The %04a is translated into a 4-digit number that encodes the SLURM_ARRAY_TASK_ID
#SBATCH --output=/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour-more_fields-1_inch-cross_val/slurm_output/out/plot_OLD_%04a_stdout.txt
#SBATCH --error=/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour-more_fields-1_inch-cross_val/slurm_output/err/plot_OLD_%04a_stderr.txt
#SBATCH --time=05:00:00
#SBATCH --job-name=plotting
#SBATCH --mail-user=tobias.schmidt@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/tgschmidt/hail/src
#SBATCH --array=0-0
#
#################################################
# $SLURM_ARRAY_TASK_ID

/home/tgschmidt/sn_env/bin/python -u plotter.py --plot_output_dir "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour-128_size-more_fields/images/test_dataset_plots" --unprocessed_labels "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour-128_size-more_fields/patches/test/unprocessed/labels/*" --hailcast_files "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour-128_size-more_fields/patches/test/hailcast/examples/*"


