#!/bin/bash

#SBATCH --partition=ai2es
# Thread count:
#SBATCH --cpus-per-task=1
# memory in MB
#SBATCH --mem=51200
# The %04a is translated into a 4-digit number that encodes the SLURM_ARRAY_TASK_ID
#SBATCH --output=/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-1_hour-1_inch-train_val_test-cross_val/slurm_output/out/min_max_multi_%04a_stdout.txt
#SBATCH --error=/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-1_hour-1_inch-train_val_test-cross_val/slurm_output/err/min_max_multi_%04a_stderr.txt
#SBATCH --time=08:00:00
#SBATCH --job-name=min_max
#SBATCH --mail-user=tobias.schmidt@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/tgschmidt/hail/src
#SBATCH --array=0-8%2
#
#################################################
num_name=$(printf "%04d" $SLURM_ARRAY_TASK_ID)

/home/tgschmidt/sn_env/bin/python -u find_min_max.py -n --examples_glob "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-1_hour-1_inch-train_val_test-cross_val/patches/multi_models/2d_model_$num_name/cv_folds/fold_0000/train/examples/*" --labels_glob "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-1_hour-1_inch-train_val_test-cross_val/patches/multi_models/2d_model_$num_name/cv_folds/fold_0000/train/labels/*" --min_maxs_dir "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-1_hour-1_inch-train_val_test-cross_val/patches/multi_models/2d_model_$num_name/cv_folds/fold_0000/mins_maxs"
/home/tgschmidt/sn_env/bin/python -u find_min_max.py -n --examples_glob "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-1_hour-1_inch-train_val_test-cross_val/patches/multi_models/2d_model_$num_name/cv_folds/fold_0001/train/examples/*" --labels_glob "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-1_hour-1_inch-train_val_test-cross_val/patches/multi_models/2d_model_$num_name/cv_folds/fold_0001/train/labels/*" --min_maxs_dir "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-1_hour-1_inch-train_val_test-cross_val/patches/multi_models/2d_model_$num_name/cv_folds/fold_0001/mins_maxs"
/home/tgschmidt/sn_env/bin/python -u find_min_max.py -n --examples_glob "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-1_hour-1_inch-train_val_test-cross_val/patches/multi_models/2d_model_$num_name/cv_folds/fold_0002/train/examples/*" --labels_glob "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-1_hour-1_inch-train_val_test-cross_val/patches/multi_models/2d_model_$num_name/cv_folds/fold_0002/train/labels/*" --min_maxs_dir "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-1_hour-1_inch-train_val_test-cross_val/patches/multi_models/2d_model_$num_name/cv_folds/fold_0002/mins_maxs"
/home/tgschmidt/sn_env/bin/python -u find_min_max.py -n --examples_glob "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-1_hour-1_inch-train_val_test-cross_val/patches/multi_models/2d_model_$num_name/cv_folds/fold_0003/train/examples/*" --labels_glob "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-1_hour-1_inch-train_val_test-cross_val/patches/multi_models/2d_model_$num_name/cv_folds/fold_0003/train/labels/*" --min_maxs_dir "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-1_hour-1_inch-train_val_test-cross_val/patches/multi_models/2d_model_$num_name/cv_folds/fold_0003/mins_maxs"
/home/tgschmidt/sn_env/bin/python -u find_min_max.py -n --examples_glob "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-1_hour-1_inch-train_val_test-cross_val/patches/multi_models/2d_model_$num_name/cv_folds/fold_0004/train/examples/*" --labels_glob "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-1_hour-1_inch-train_val_test-cross_val/patches/multi_models/2d_model_$num_name/cv_folds/fold_0004/train/labels/*" --min_maxs_dir "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-1_hour-1_inch-train_val_test-cross_val/patches/multi_models/2d_model_$num_name/cv_folds/fold_0004/mins_maxs"


