#!/bin/bash
#SBATCH -p ai2es_v100
#SBATCH --nodes=1-1
#SBATCH -n 1
#SBATCH --mem-per-cpu=1000
#SBATCH --time=00:30:00
#SBATCH --job-name="unet_predict"
#SBATCH --mail-user=lydiaks2@illinois.edu
#SBATCH --mail-type=ALL
#SBATCH --mail-type=END
#SBATCH --output=unet_test_data/R-%x.%j.out
#SBATCH --error=unet_test_data/R-%x.%j.err

#Running using Modules
echo "loading modules"
module load TensorFlow/2.4.1-fosscuda-2020a-Python-3.8.2
#echo "installing packages"
pip install keras-unet-collection
pip install Pillow
pip install xarray
pip install netCDF4
pip install dask
pip install scikit-learn

#echo "loading python"
source .bashrc
bash

#echo "running python"
python case_study_predict.py