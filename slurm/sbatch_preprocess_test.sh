#!/bin/bash

#SBATCH --partition=ai2es
# Thread count:
#SBATCH --cpus-per-task=1
# memory in MB
#SBATCH --mem=15360
# The %04a is translated into a 4-digit number that encodes the SLURM_ARRAY_TASK_ID
#SBATCH --output=/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/slurm_output/out/prep_test_refl_%04a_stdout.txt
#SBATCH --error=/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/slurm_output/err/prep_test_refl_%04a_stderr.txt
#SBATCH --time=05:00:00
#SBATCH --job-name=test_preprocess
#SBATCH --mail-user=tobias.schmidt@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/tgschmidt/hail/src
#SBATCH --array=0-11
#
#################################################
# $SLURM_ARRAY_TASK_ID

/home/tgschmidt/sn_env/bin/python -u preprocessor.py -s -d -m -t -n -u --run_num $SLURM_ARRAY_TASK_ID --examples "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/unprocessed_no_shuffle/examples/*" --labels "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/unprocessed_no_shuffle/labels/*" --output_ds_dir "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/no_shuffle/processed" --label_class_name_str "MESH_class_bin_severe" --min_max_dir "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/mins_maxs_gridrad_refl" --n_parallel_runs 12


