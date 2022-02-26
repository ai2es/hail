#!/bin/bash
#SBATCH -p ai2es_v100
#SBATCH --nodes=1
#SBATCH -n 1
#SBATCH --mem-per-cpu=1000
#SBATCH --time=00:05:00
#SBATCH --job-name="test_tensorflowgpu"
#SBATCH --mail-user=robjk3-22@rhodes.edu
#SBATCH --mail-type=ALL
#SBATCH --mail-type=END
#SBATCH --output=unet_test_data/R-%x.%j.out
#SBATCH --error=unet_test_data/R-%x.%j.err

module load TensorFlow/2.4.1-fosscuda-2020a-Python-3.8.2 

python testing_gpu.py