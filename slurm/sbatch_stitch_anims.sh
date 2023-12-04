#!/bin/bash

#SBATCH --partition=ai2es
# Thread count:
#SBATCH --cpus-per-task=8
# memory in MB
#SBATCH --mem=16384
# The %04a is translated into a 4-digit number that encodes the SLURM_ARRAY_TASK_ID
#SBATCH --output=/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/slurm_output/out/stitching_%04a_stdout.txt
#SBATCH --error=/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/slurm_output/err/stitching_%04a_stderr.txt
#SBATCH --time=02:00:00
#SBATCH --job-name=stitching
#SBATCH --mail-user=tobias.schmidt@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/tgschmidt/hail/src
#SBATCH --array=0-0
#
#################################################
# $SLURM_ARRAY_TASK_ID

/home/tgschmidt/sn_env/bin/python -u plotter.py -d -p --stitched_domain_file "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/animations/averaged_anims/20170518/stitched_domain/0.nc" --predictions_paths "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/animations/averaged_anims/20170518/0/predictions/y_hats.nc" --unprocessed_examples "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/animations/averaged_anims/20170518/0/unprocessed/examples/0000.nc" --unprocessed_labels "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/animations/averaged_anims/20170518/0/unprocessed/labels/0000.nc"
/home/tgschmidt/sn_env/bin/python -u plotter.py -d -p --stitched_domain_file "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/animations/averaged_anims/20170518/stitched_domain/1.nc" --predictions_paths "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/animations/averaged_anims/20170518/1/predictions/y_hats.nc" --unprocessed_examples "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/animations/averaged_anims/20170518/1/unprocessed/examples/0000.nc" --unprocessed_labels "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/animations/averaged_anims/20170518/1/unprocessed/labels/0000.nc"
/home/tgschmidt/sn_env/bin/python -u plotter.py -d -p --stitched_domain_file "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/animations/averaged_anims/20170518/stitched_domain/2.nc" --predictions_paths "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/animations/averaged_anims/20170518/2/predictions/y_hats.nc" --unprocessed_examples "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/animations/averaged_anims/20170518/2/unprocessed/examples/0000.nc" --unprocessed_labels "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/animations/averaged_anims/20170518/2/unprocessed/labels/0000.nc"
/home/tgschmidt/sn_env/bin/python -u plotter.py -d -p --stitched_domain_file "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/animations/averaged_anims/20170518/stitched_domain/3.nc" --predictions_paths "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/animations/averaged_anims/20170518/3/predictions/y_hats.nc" --unprocessed_examples "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/animations/averaged_anims/20170518/3/unprocessed/examples/0000.nc" --unprocessed_labels "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/animations/averaged_anims/20170518/3/unprocessed/labels/0000.nc"



