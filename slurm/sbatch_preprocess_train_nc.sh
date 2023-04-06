#!/bin/bash

#SBATCH --partition=ai2es
# Thread count:
#SBATCH --cpus-per-task=3
# memory in MB
#SBATCH --mem=81920
# The %04a is translated into a 4-digit number that encodes the SLURM_ARRAY_TASK_ID
#SBATCH --output=/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour-more_fields-1_inch-cross_val/slurm_output/out/prep_trainlog_%04a_stdout.txt
#SBATCH --error=/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour-more_fields-1_inch-cross_val/slurm_output/err/prep_trainlog_%04a_stderr.txt
#SBATCH --time=03:00:00
#SBATCH --job-name=train_preprocess
#SBATCH --mail-user=tobias.schmidt@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/tgschmidt/hail/src
#SBATCH --array=0-29
#
#################################################
# $SLURM_ARRAY_TASK_ID

/home/tgschmidt/sn_env/bin/python -u preprocessor.py -m -t -n -s --run_num $SLURM_ARRAY_TASK_ID --examples "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour-more_fields-1_inch-cross_val/patches/cv_folds/fold_0000/train/examples/*" --labels "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour-more_fields-1_inch-cross_val/patches/cv_folds/fold_0000/train/labels/*" --output_ds_dir "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour-more_fields-1_inch-cross_val/patches/cv_folds/fold_0000/log_reg_files/train" --n_parallel_runs 30


