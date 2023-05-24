#!/bin/bash

#SBATCH --partition=ai2es
# Thread count:
#SBATCH --cpus-per-task=4
# memory in MB
#SBATCH --mem=30720
# The %04a is translated into a 4-digit number that encodes the SLURM_ARRAY_TASK_ID
#SBATCH --output=/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/slurm_output/out/grid_gridrad_1_%04a_stdout.txt
#SBATCH --error=/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/slurm_output/err/grid_gridrad_1_%04a_stderr.txt
#SBATCH --time=infinite
#SBATCH --job-name=21_grid_gridrad
#SBATCH --mail-user=tobias.schmidt@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/tgschmidt/hail/src
#SBATCH --array=0-0
#
#################################################
# $SLURM_ARRAY_TASK_ID

/home/tgschmidt/sn_env/bin/python -u gridrad_gridder.py --input_year_dir_glob "/ourdisk/hpc/ai2es/gridrad_severe/volumes/2021/*" --output_dir "/ourdisk/hpc/ai2es/severe_nowcasting/gridrad_gridded/2021"