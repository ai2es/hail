#!/bin/bash

#SBATCH --partition=ai2es
#SBATCH --cpus-per-task=12
# memory in MB
#SBATCH --mem=400000
#SBATCH --nodelist=c829
#SBATCH --exclusive
# The %04a is translated into a 4-digit number that encodes the SLURM_ARRAY_TASK_ID
#SBATCH --output=/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour-more_fields-1_inch-cross_val/slurm_output/out/predict_3D_%04a_stdout.txt
#SBATCH --error=/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour-more_fields-1_inch-cross_val/slurm_output/err/predict_3D_%04a_stderr.txt
#SBATCH --time=05:00:00
#SBATCH --job-name=predict_3D
#SBATCH --mail-user=tobias.schmidt@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/tgschmidt/hail/src
#SBATCH --array=0-0
#
#################################################
# $SLURM_ARRAY_TASK_ID

/home/tgschmidt/sn_env/bin/python -u predictor.py -u --run_num $SLURM_ARRAY_TASK_ID --single_checkpoint "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour-more_fields-1_inch-cross_val/saved_models_fold_0000/checkpoints/36" --predictions_outfile "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour-more_fields-1_inch-cross_val/patches/test/predictions/y_hats_flat.nc" --examples "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour-more_fields-1_inch-cross_val/patches/test/processed_flattened/examples/*" --labels "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour-more_fields-1_inch-cross_val/patches/test/processed_flattened/labels/*"


