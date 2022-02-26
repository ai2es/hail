import xarray as xr
import numpy as np
import glob
import gzip
import shutil 
import os
from scipy import interpolate
import time
import sys
import copy
import pandas as pd

#time the output generation
tic = time.perf_counter()

#Find all of the days that we have mesh data for
all_mesh_dirs = glob.glob('/ourdisk/hpc/ai2es/hail/mrms/mrms/NCEP/*/MESH/')
all_mesh_dirs.sort()

#load in the latitudes and longitudes for interpolation
#this can be any hrrr file. Doesn't matter about the data/time
hrrr = xr.open_dataset('../../ourdisk/hpc/ai2es/hail/ml_input/20160401/20160401_0000.nc')
lats = hrrr.latitude
lons = hrrr.longitude

#Index primer indicates the day of data that we are looking at in this particuar run of this code
index_primer = int(sys.argv[1])
df = pd.to_datetime(pd.read_csv('/home/randychase/Order_to_process.csv',header=0,index_col=0).values[:,0])
this_mesh_dir = df[index_primer].strftime('/ourdisk/hpc/ai2es/hail/mrms/mrms/NCEP/%Y%m%d/MESH/')
filenames = glob.glob(this_mesh_dir+'*00.50*.grib2.gz')

#pull out the date of the data that we are using
date = df[index_primer].strftime('%Y%m%d')

#read in all the filenames
filenames = glob.glob(f"../../ourdisk/hpc/ai2es/hail/mrms/mrms/NCEP/%s/MESH/*00.50*.grib2.gz" %(date))
filenames.sort()




#declare all the filepaths used in this python file
path_to_mrms = "../../ourdisk/hpc/ai2es/hail/mrms/mrms/NCEP"
path_to_output = "../../ourdisk/hpc/ai2es/hail/ml_output_2"
old_path_to_output = "../../ourdisk/hpc/ai2es/hail/ml_output"
tmp_filepath = '/scratch/lydiaks2/tmp_mrms/' + date + '/'
if not os.path.exists(tmp_filepath):
        os.mkdir(tmp_filepath)


#if the file was already unzipped
if(len(filenames) == 0):
    #read in all the filenames
    #filenames = glob.glob(f"../../ourdisk/hpc/ai2es/hail/mrms/mrms/NCEP/%s/MESH/*00.50*.grib2" %(date))
    filenames = glob.glob(this_mesh_dir+'*00.50*.grib2')
    #there is data from this day and the prior day in the folder
    yesterdate = (filenames[0])[-21:-13]
    #open each file and copy it to the temporary directory
    for filename in filenames:
        with open(filename, 'rb') as f_in:
            tmp_filename = tmp_filepath + filename[-37:-6] + '_tmp.grib2'
            with open(tmp_filename, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)        
else: #the files were zipped
    #there is data from this day and the prior day in the folder
    yesterdate = (filenames[0])[-24:-16]
    for filename in filenames:
        #open each file, unzip it, and copy it to the temporary directory
        with gzip.open(filename, 'rb') as f_in:
           tmp_filename = tmp_filepath + filename[-40:-9] + '_tmp.grib2'
           with open(tmp_filename, 'wb') as f_out:
               shutil.copyfileobj(f_in, f_out)


#separate the data into hours because we are combining each hour
yesterhours = []
hours = []
for hr in range(23):
    #read in the files and sort them into their correct place in the hours array
    files = glob.glob(f"%s*00.50*%s-%02d*.grib2" %(tmp_filepath,date,hr))
    files.sort()
    hours.append(files)
    
    yesterfiles = glob.glob(f"%s*00.50*%s-%02d*.grib2" %(tmp_filepath,yesterdate,hr))
    yesterfiles.sort()
    hours.append(yesterfiles)



#create the data for each hour
for hour in hours:
    if len(hour) == 0:
        continue

    #define the date and hour
    date = (hour[0])[-25:-17]
    hr = (hour[0])[-16:-14]
    
    # I want to check to see if the processed file already exists, to save computational time
    if os.path.exists("%s/%s/%s_%s00.nc" % (old_path_to_output,date,date,hr)):
        print("PATH EXISTS ----- CONTINUING ----------")
        continue    
    
    
    
    #we have had some errors that kill the script for a certain hour. To avoid this issue, we will use 'exception' behavior. 
    try:    
        #create a DataArray of this hour
        ds = xr.open_mfdataset(hour, concat_dim='valid_time',combine='nested', parallel=True)#.compute()
    except:
        print('ERROR WITH LOADING. Please check these files at a later date \n')
        for m in hour:
            print(m + '\n')
        continue 
    
    
    
    #We need to aggregate the dataset - 90th percentile in the hour of the non-zero values
    #convert zeros and negatives to nans
    posmesh = ds.where(ds.unknown.values > 0)#.compute()
    
    
    #Reallocate how the CPUs are splitting up this dataset because they were split up on time, but we are combining by time
    posmesh = posmesh.chunk(chunks={'valid_time':-1,'latitude':100,'longitude':100})#.compute()
    
    
    #aggregate over time to the 90th percentile of non-nan values
    withnans = posmesh.quantile(.9,dim='valid_time',skipna = True).compute()
    
    
    #convert nans back to zero for interpolation
    withnans.unknown.values = np.where(np.isnan(withnans.unknown.values), 0, withnans.unknown.values)
    
    
    # Evaluate the interpolation to create an array on the same grid as the HRRR
    mesh_regridded = withnans.interp(longitude=lons,latitude=lats).compute()
    new_mrms = mesh_regridded.unknown.values
    mesh_90 = copy.deepcopy(mesh_regridded.unknown)
    
    
    #We are doing a classification problem, label the data so that it is:
    #For the regression problem, take out this chunk of code
    # 0 = no hail
    # 1 = non-severe hail
    # 2 = severe hail
    # 3 = significant severe hail
    labels = np.zeros((1059,1799))
    labels = np.where(new_mrms == 0, 0, labels)
    labels = np.where(np.logical_and(0 < new_mrms, new_mrms <= 25), 1, labels)
    labels = np.where(np.logical_and(25 < new_mrms, new_mrms <= 50), 2, labels)
    labels = np.where(new_mrms > 50, 3, labels)
    mesh_regridded.unknown.values = labels
    
    
    
    
    #if there is not already a directory where we are putting the output, make it
    if not os.path.exists(os.path.join(path_to_output, date)):
        os.mkdir(os.path.join(path_to_output, date))
    
    
    #save using netcdf
    mesh_regridded = mesh_regridded.rename({'unknown': "mesh"})
    mesh_regridded['mesh_90'] = mesh_90 #this will name it mesh_90 in the dataset 
    mesh_regridded.to_netcdf("%s/%s/%s_%s00.nc" % (path_to_output,date,date,hr), compute=True)
    mesh_regridded.close() #ADDED THIS
    
    '''
    #Delete the temporary files
    toRemove = glob.glob('%s*%s-%s*' % (tmp_filepath,date,hr))
    for file in toRemove:
        if(os.path.exists(f'%s' % (file))):
            os.remove(f'%s' % (file))
    '''

'''
#delete the temporary directory
if(os.path.exists(tmp_filepath)):
    os.rmdir(tmp_filepath[:-1])
'''
    
    
toc = time.perf_counter()
print(f"Total time = {toc - tic:0.4f} seconds")