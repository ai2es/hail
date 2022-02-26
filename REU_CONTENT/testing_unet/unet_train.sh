#!/bin/bash
#SBATCH -p debug_5min
#SBATCH --nodes=1-1
#SBATCH -n 4
#SBATCH --mem-per-cpu=1000
#SBATCH --time=00:05:00
#SBATCH --job-name="unet_train"
#SBATCH --mail-user=lydiaks2@illinois.edu
#SBATCH --mail-type=ALL

#Running using Modules
echo "loading modules"
# module load GCCcore/6.4.0
module load TensorFlow/2.4.1-fosscuda-2020a-Python-3.8.2 
#install 'togo' packages, it will install wheels to use in your local cache, ~/.local/
pip install keras-unet-collection
pip install Pillow
pip install xarray
pip install netCDF4

#source ~/.bashrc
#bash
#conda activate hagelslag

python unet_train.py






