#!/bin/bash

#SBATCH --partition=ai2es
# Thread count:
#SBATCH --cpus-per-task=8
# memory in MB
#SBATCH --mem=32768
# The %04a is translated into a 4-digit number that encodes the SLURM_ARRAY_TASK_ID
#SBATCH --output=/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/slurm_output/out/CV_multimodel_%04a_stdout.txt
#SBATCH --error=/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/slurm_output/err/CV_multimodel_%04a_stderr.txt
#SBATCH --time=72:00:00
#SBATCH --job-name=2D_cross_val
#SBATCH --mail-user=tobias.schmidt@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/tgschmidt/hail/src
#SBATCH --array=0-8
#
#################################################
num_name=$(printf "%04d" $SLURM_ARRAY_TASK_ID)

/home/tgschmidt/sn_env/bin/python -u splitter.py --examples_glob "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/multi_models/2d_model_$num_name/trainval/examples/*" --labels_glob "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/multi_models/2d_model_$num_name/trainval/labels/*" --fold_path "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/multi_models/2d_model_$num_name/cv_folds" --num_output_files 30 --n_folds 5