#!/bin/bash

#SBATCH --partition=ai2es
# Thread count:
#SBATCH --cpus-per-task=12
# memory in MB
#SBATCH --mem=307200
# The %04a is translated into a 4-digit number that encodes the SLURM_ARRAY_TASK_ID
#SBATCH --output=/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour-more_fields-1_inch-cross_val/slurm_output/out/log_reg_pred_%04a_stdout.txt
#SBATCH --error=/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour-more_fields-1_inch-cross_val/slurm_output/err/log_reg_pred_%04a_stderr.txt
#SBATCH --time=05:00:00
#SBATCH --job-name=log_reg
#SBATCH --mail-user=tobias.schmidt@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/tgschmidt/hail/src
#SBATCH --array=0-0
#
#################################################
# $SLURM_ARRAY_TASK_ID

/home/tgschmidt/sn_env/bin/python -u log_reg.py -p --n_cpus 12 --examples "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour-more_fields-1_inch-cross_val/patches/cv_folds/fold_0000/log_reg_files/train/examples/*" --labels "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour-more_fields-1_inch-cross_val/patches/cv_folds/fold_0000/log_reg_files/train/labels/*" --val_examples "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour-more_fields-1_inch-cross_val/patches/cv_folds/fold_0000/log_reg_files/val/examples/*" --val_labels "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour-more_fields-1_inch-cross_val/patches/cv_folds/fold_0000/log_reg_files/val/labels/*" --saved_model_path "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour-more_fields-1_inch-cross_val/patches/cv_folds/fold_0000/log_reg_files/saved_models/log_reg.sav" --test_examples "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour-more_fields-1_inch-cross_val/patches/test/processed/examples/*" --predictions_outfile "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour-more_fields-1_inch-cross_val/patches/cv_folds/fold_0000/log_reg_files/predictions/y_hats.nc"


