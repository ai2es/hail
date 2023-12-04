#!/bin/bash

#SBATCH --partition=ai2es
#SBATCH --cpus-per-task=6
# memory in MB
#SBATCH --mem=51200
# The %04a is translated into a 4-digit number that encodes the SLURM_ARRAY_TASK_ID
#SBATCH --output=/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/slurm_output/out/pred_%04a_stdout.txt
#SBATCH --error=/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/slurm_output/err/pred_%04a_stderr.txt
#SBATCH --time=10:00:00
#SBATCH --job-name=pred
#SBATCH --mail-user=tobias.schmidt@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/tgschmidt/hail/src
#SBATCH --array=0-0
#
#################################################
# $SLURM_ARRAY_TASK_ID

/home/tgschmidt/sn_env/bin/python -u predictor.py -u --ens_member 1 --run_num $SLURM_ARRAY_TASK_ID --single_checkpoint "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/saved_models_gaus/fold_0000/checkpoints/46" --predictions_outfile "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/animations/averaged_anims/20190528/0/predictions/y_hats.nc" --examples "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/animations/averaged_anims/20190528/0/processed/examples/*" --labels "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/animations/averaged_anims/20190528/0/processed/labels/*"
/home/tgschmidt/sn_env/bin/python -u predictor.py -u --ens_member 1 --run_num $SLURM_ARRAY_TASK_ID --single_checkpoint "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/saved_models_gaus/fold_0000/checkpoints/46" --predictions_outfile "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/animations/averaged_anims/20190528/1/predictions/y_hats.nc" --examples "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/animations/averaged_anims/20190528/1/processed/examples/*" --labels "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/animations/averaged_anims/20190528/1/processed/labels/*"
/home/tgschmidt/sn_env/bin/python -u predictor.py -u --ens_member 1 --run_num $SLURM_ARRAY_TASK_ID --single_checkpoint "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/saved_models_gaus/fold_0000/checkpoints/46" --predictions_outfile "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/animations/averaged_anims/20190528/2/predictions/y_hats.nc" --examples "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/animations/averaged_anims/20190528/2/processed/examples/*" --labels "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/animations/averaged_anims/20190528/2/processed/labels/*"
/home/tgschmidt/sn_env/bin/python -u predictor.py -u --ens_member 1 --run_num $SLURM_ARRAY_TASK_ID --single_checkpoint "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/saved_models_gaus/fold_0000/checkpoints/46" --predictions_outfile "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/animations/averaged_anims/20190528/3/predictions/y_hats.nc" --examples "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/animations/averaged_anims/20190528/3/processed/examples/*" --labels "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/animations/averaged_anims/20190528/3/processed/labels/*"


