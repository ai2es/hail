import xarray as xr
import numpy as np
import glob
import gzip
import shutil 
import os
from scipy import interpolate
import time
import sys

#Find all of the days that we have mesh data for
all_mesh_dirs = glob.glob('/ourdisk/hpc/ai2es/hail/mrms/mrms/NCEP/*/MESH/')
all_mesh_dirs.sort()


#Index primer indicates the day of data that we are looking at in this particuar run of this code
index_primer = int(sys.argv[1])
#find which day corresponds to index_primer
this_mesh_dir = all_mesh_dirs[index_primer]

#load in the latitudes and longitudes for interpolation
hrrr = xr.open_dataset('../../ourdisk/hpc/ai2es/hail/ml_input/20190501/20190501_0000.nc')
lats = hrrr.latitude
lons = hrrr.longitude

#declare all the filepaths used in this python file
path_to_mrms = "../../ourdisk/hpc/ai2es/hail/mrms/mrms/NCEP"
path_to_output = "../../ourdisk/hpc/ai2es/hail/ml_probabilities"
tmp_filepath = 'scratch/tmp_mrms/'

#pull out the date of the data that we are using
date = this_mesh_dir[-14:-6]

#MOVED mkdir to the end

#read in all the filenames
filenames = glob.glob(f"../../ourdisk/hpc/ai2es/hail/mrms/mrms/NCEP/%s/MESH/*00.50*.grib2.gz" %(date))
filenames.sort()

#if the file was already unzipped
if(len(filenames) == 0):
    #read in all the filenames
    filenames = glob.glob(f"../../ourdisk/hpc/ai2es/hail/mrms/mrms/NCEP/%s/MESH/*00.50*.grib2" %(date))
    filenames.sort()
    #there is data from this day and the prior day in the folder
    yesterdate = (filenames[0])[-21:-13] #ADDED THIS
    #open each file and copy it to the temporary directory
    for filename in filenames:
        with open(filename, 'rb') as f_in:
            tmp_filename = tmp_filepath + filename[-37:-6] + '_tmp.grib2'
            with open(tmp_filename, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)        
else: #the files were zipped
    #there is data from this day and the prior day in the folder
    yesterdate = (filenames[0])[-24:-16] #ADDED THIS
    for filename in filenames:
        #open each file, unzip it, and copy it to the temporary directory
        with gzip.open(filename, 'rb') as f_in:
           tmp_filename = tmp_filepath + filename[-40:-9] + '_tmp.grib2'
           with open(tmp_filename, 'wb') as f_out:
               shutil.copyfileobj(f_in, f_out)
               
#separate the data into hours because we are combining each hour
yesterhours = [] #ADDED THIS
hours = []
for hr in range(24):
    #read in the files and sort them into their correct place in the hours array
    files = glob.glob(f"%s*00.50*%s-%02d*.grib2" %(tmp_filepath,date,hr)) #CHANGED THIS
    files.sort()
    hours.append(files)
    
    yesterfiles = glob.glob(f"%s*00.50*%s-%02d*.grib2" %(tmp_filepath,yesterdate,hr)) #ADDED THIS
    yesterfiles.sort() #ADDED THIS
    hours.append(yesterfiles) #ADDED THIS


#create the data for each hour
for hour in hours:
    if len(hour) == 0:
        continue

    #REMOVED hr
    
    #create a DataArray of this hour
    ds = xr.open_mfdataset(hour, concat_dim='valid_time',combine='nested', parallel=True)
    
    
    #We need to aggregate the dataset - 90th percentile in the hour of the non-zero values
    #convert zeros and negatives to nans
    posmesh = ds.where(ds.unknown.values > 0)
    
    
    #Reallocate how the CPUs are splitting up this dataset because they were split up on time, but we are combining by time
    posmesh = posmesh.chunk(chunks={'valid_time':-1,'latitude':100,'longitude':100})
    
    
    #aggregate over time to the 90th percentile of non-nan values
    withnans = posmesh.quantile(.9,dim='valid_time',skipna = True).compute()
    
    
    #convert nans back to zero for interpolation
    withnans.unknown.values = np.where(np.isnan(withnans.unknown.values), 0, withnans.unknown.values)
    
    
    # Evaluate the interpolation to create an array on the same grid as the HRRR
    mesh_regridded = posmesh.interp(longitude=lons,latitude=lats).compute()
    new_mrms = mesh_regridded.unknown.values
    



