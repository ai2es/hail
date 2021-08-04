#!/bin/bash
#SBATCH -p debug_5min
#SBATCH --nodes=1-1
#SBATCH -n 10
#SBATCH --mem-per-cpu=1000
#SBATCH --time=00:05:00
#SBATCH --job-name="runfile"
#SBATCH --mail-user=robjk3-22@rhodes.edu
#SBATCH --mail-type=ALL
#SBATCH --mail-type=END
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.err

import numpy as np
import pygrib
from tqdm import tqdm

import netCDF4
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from os.path import exists

import cartopy as cp
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import cartopy.feature as cfeature
import glob

mrms_files = glob.glob("c:/Users/jorda/Downloads/MESH/*.grib2")
mrms_files.sort()
file_list = []
for file in tqdm(mrms_files[90:120]):
     file_list.append(xr.open_dataset(file))
ds = xr.concat(file_list, dim="valid_time")

from dask.diagnostics import ProgressBar
res = ds.unknown.quantile(0.9, dim="valid_time", skipna=True)
with ProgressBar():
     out = res.compute()

v90.to_netcdf("conv_test/MESH_HRRR_20160413_03z.nc")
