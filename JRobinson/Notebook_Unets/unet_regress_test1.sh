#!/bin/bash
#SBATCH -p normal
#SBATCH --nodes=1-1
#SBATCH -n 4
#SBATCH --mem-per-cpu=1000
#SBATCH --time=00:25:00
#SBATCH --job-name="unet_regress_test1"
#SBATCH --mail-user=robjk3_22@rhodes.edu
#SBATCH --mail-type=ALL
#SBATCH --mail-type=END
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.err

#Running using Modules
echo "loading modules"
# module load GCCcore/6.4.0
module load TensorFlow/2.4.1-fosscuda-2020a-Python-3.8.2 
#install 'togo' packages, it will install wheels to use in your local cache, ~/.local/
echo "installing packages"
pip install keras-unet-collection
pip install Pillow
pip install xarray
pip install netCDF4
pip install dask

source /home/jordot45/.bashrc
bash 
conda activate hagelslag
python unet_regress_test1.py