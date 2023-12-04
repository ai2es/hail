#!/bin/bash

#SBATCH --partition=ai2es
# Thread count:
#SBATCH --cpus-per-task=1
# memory in MB
#SBATCH --mem=81920
# The %04a is translated into a 4-digit number that encodes the SLURM_ARRAY_TASK_ID
#SBATCH --output=/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/slurm_output/out/prep_val_multi_gaus_fold_0000_%04a_stdout.txt
#SBATCH --error=/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/slurm_output/err/prep_val_multi_gaus_fold_0000_%04a_stderr.txt
#SBATCH --time=15:00:00
#SBATCH --job-name=mul_val_preprocess
#SBATCH --mail-user=tobias.schmidt@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/tgschmidt/hail/src
#SBATCH --array=0-29%6
#
#################################################
# $SLURM_ARRAY_TASK_ID

/home/tgschmidt/sn_env/bin/python -u preprocessor.py -d -m -n -u -g --run_num $SLURM_ARRAY_TASK_ID --examples "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/multi_models/2d_model_0000/cv_folds/fold_0000/val/examples/*" --labels "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/multi_models/2d_model_0000/cv_folds/fold_0000/val/labels/*" --output_ds_dir "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/multi_models/2d_model_0000/cv_folds/fold_0000/val/tf_datasets" --n_parallel_runs 30
/home/tgschmidt/sn_env/bin/python -u preprocessor.py -d -m -n -u -g --run_num $SLURM_ARRAY_TASK_ID --examples "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/multi_models/2d_model_0001/cv_folds/fold_0000/val/examples/*" --labels "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/multi_models/2d_model_0001/cv_folds/fold_0000/val/labels/*" --output_ds_dir "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/multi_models/2d_model_0001/cv_folds/fold_0000/val/tf_datasets" --n_parallel_runs 30
/home/tgschmidt/sn_env/bin/python -u preprocessor.py -d -m -n -u -g --run_num $SLURM_ARRAY_TASK_ID --examples "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/multi_models/2d_model_0002/cv_folds/fold_0000/val/examples/*" --labels "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/multi_models/2d_model_0002/cv_folds/fold_0000/val/labels/*" --output_ds_dir "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/multi_models/2d_model_0002/cv_folds/fold_0000/val/tf_datasets" --n_parallel_runs 30
/home/tgschmidt/sn_env/bin/python -u preprocessor.py -d -m -n -u -g --run_num $SLURM_ARRAY_TASK_ID --examples "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/multi_models/2d_model_0003/cv_folds/fold_0000/val/examples/*" --labels "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/multi_models/2d_model_0003/cv_folds/fold_0000/val/labels/*" --output_ds_dir "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/multi_models/2d_model_0003/cv_folds/fold_0000/val/tf_datasets" --n_parallel_runs 30
/home/tgschmidt/sn_env/bin/python -u preprocessor.py -d -m -n -u -g --run_num $SLURM_ARRAY_TASK_ID --examples "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/multi_models/2d_model_0004/cv_folds/fold_0000/val/examples/*" --labels "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/multi_models/2d_model_0004/cv_folds/fold_0000/val/labels/*" --output_ds_dir "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/multi_models/2d_model_0004/cv_folds/fold_0000/val/tf_datasets" --n_parallel_runs 30
/home/tgschmidt/sn_env/bin/python -u preprocessor.py -d -m -n -u -g --run_num $SLURM_ARRAY_TASK_ID --examples "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/multi_models/2d_model_0005/cv_folds/fold_0000/val/examples/*" --labels "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/multi_models/2d_model_0005/cv_folds/fold_0000/val/labels/*" --output_ds_dir "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/multi_models/2d_model_0005/cv_folds/fold_0000/val/tf_datasets" --n_parallel_runs 30
/home/tgschmidt/sn_env/bin/python -u preprocessor.py -d -m -n -u -g --run_num $SLURM_ARRAY_TASK_ID --examples "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/multi_models/2d_model_0006/cv_folds/fold_0000/val/examples/*" --labels "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/multi_models/2d_model_0006/cv_folds/fold_0000/val/labels/*" --output_ds_dir "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/multi_models/2d_model_0006/cv_folds/fold_0000/val/tf_datasets" --n_parallel_runs 30
/home/tgschmidt/sn_env/bin/python -u preprocessor.py -d -m -n -u -g --run_num $SLURM_ARRAY_TASK_ID --examples "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/multi_models/2d_model_0007/cv_folds/fold_0000/val/examples/*" --labels "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/multi_models/2d_model_0007/cv_folds/fold_0000/val/labels/*" --output_ds_dir "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/multi_models/2d_model_0007/cv_folds/fold_0000/val/tf_datasets" --n_parallel_runs 30
/home/tgschmidt/sn_env/bin/python -u preprocessor.py -d -m -n -u -g --run_num $SLURM_ARRAY_TASK_ID --examples "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/multi_models/2d_model_0008/cv_folds/fold_0000/val/examples/*" --labels "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/multi_models/2d_model_0008/cv_folds/fold_0000/val/labels/*" --output_ds_dir "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/multi_models/2d_model_0008/cv_folds/fold_0000/val/tf_datasets" --n_parallel_runs 30


