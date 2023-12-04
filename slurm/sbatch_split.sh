#!/bin/bash

#SBATCH --partition=ai2es
# Thread count:
#SBATCH --cpus-per-task=3
# memory in MB
#SBATCH --mem=102400
# The %04a is translated into a 4-digit number that encodes the SLURM_ARRAY_TASK_ID
#SBATCH --output=/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/slurm_output/out/CV_single_model_%04a_stdout.txt
#SBATCH --error=/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/slurm_output/err/CV_single_model_%04a_stderr.txt
#SBATCH --time=10:00:00
#SBATCH --job-name=CV_single_model
#SBATCH --mail-user=tobias.schmidt@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/tgschmidt/hail/src
#SBATCH --array=0-0
#
#################################################
/home/tgschmidt/sn_env/bin/python -u splitter.py --examples_glob "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/single_2d_unet_model/trainval/examples/*" --labels_glob "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/single_2d_unet_model/trainval/labels/*" --fold_path "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/single_2d_unet_model/cv_folds" --num_output_files 30 --n_folds 5

# /home/tgschmidt/sn_env/bin/python -u splitter.py --examples_glob "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/trainvaltest/examples/*" --labels_glob "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/trainvaltest/labels/*" --first_ds_output "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/trainval" --second_ds_output "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/unprocessed" --n_folds 5