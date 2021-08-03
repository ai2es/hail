
import numpy as np
import pygrib
from tqdm import tqdm

import netCDF4
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from os.path import exists

#import cartopy as cp
#import cartopy.crs as ccrs
#import cartopy.io.shapereader as shpreader
#import cartopy.feature as cfeature
import glob

mrms_files = glob.glob("/ourdisk/hpc/ai2es/hail/mrms/mrms/NCEP/20160413/MESH/*.grib2")
mrms_files.sort()

#Creating temporary file
tmp_filepath = 'scratch/tmp_mrms/'
#pull out the date of the data that we are using
date = this_mesh_dir[-14:-6]


file_list = []
hours = []
for hr in range(24):
    #read in the files and sort them into their correct place in the hours array
    files = glob.glob(f"%s*00.50*%s-%02d*.grib2" %(tmp_filepath,date,hr)) #CHANGED THIS
    files.sort()
    hours.append(files)
ds = xr.concat(file_list, dim="valid_time")

ds = ds.where(ds.unknown >= 0)
# Grab the 90th percentile values of non-zero MESH values
from dask.diagnostics import ProgressBar
res = ds.unknown.quantile(0.9, dim="valid_time", skipna=True)
with ProgressBar():
     out = res.compute()
v90 = res

v90.to_netcdf("./MESH_HRRR_20160413_03z.nc")