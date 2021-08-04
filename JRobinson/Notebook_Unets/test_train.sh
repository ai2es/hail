#!/bin/bash
#SBATCH -p debug
#SBATCH --nodes=1-1
#SBATCH -n 4
#SBATCH --exclusive
#SBATCH --mem-per-cpu=1000
#SBATCH --time=00:30:00
#SBATCH --job-name="unet_train"
#SBATCH --mail-user=robjk3-22@rhodes.edu
#SBATCH --mail-type=ALL
#SBATCH --mail-type=END
#SBATCH --output=/ourdisk/hpc/ai2es/jRobinson/R-%x.%j.out
#SBATCH --error=/ourdisk/hpc/ai2es/jRobinson/R-%x.%j.err


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

echo "loading python"
source .bashrc
bash
conda activate hagelslag
echo "running python"
python test_train.py