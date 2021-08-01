from keras_unet_collection import models, losses
import tensorflow as tf
from tensorflow import keras
import xarray as xr
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import csv
import glob
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

#define all the file paths
#path to patches is the location all the patch files are stored
path_to_patches = '/ourdisk/hpc/ai2es/hail/images_for_learning/*'
#checkpoint path is where the model will be stored
checkpoint_path = 'unet_data/final_cce_2000_1en4/'
#where to save metadata?
metadata_outfile_name = checkpoint_path + 'metadata_predict.csv'
#where to save predictions
predictions_outfile_name = checkpoint_path + 'predictions.nc'

#read in the data
infiles = glob.glob(path_to_patches)
#open all the data in one DataArray
ds = xr.open_mfdataset(infiles, concat_dim='n_samples',combine='nested', parallel=True, engine='netcdf4')
#isolate only the testing data
ds = ds.where(ds.time > np.datetime64('2019-07-01'),drop=True)#.dropna(dim='n_samples')


#Separate the data array into ml input, ml output and metadata
output_ds = ds.mesh
output_array = output_ds.values

#read out the metadata
meshfrac = ds.meshfrac.values
time = ds.time.values
lon = ds.lon.values
lat = ds.lat.values
mesh_90_mean = ds.mesh_90_mean.values
mesh_90_std = ds.mesh_90_std.values
mesh_90_percentiles = ds.mesh_90_percentiles.values

#drop the metadata from the DataArray
to_drop = ['mesh','mesh_90','meshfrac','time','lon','lat','mesh_90_mean','mesh_90_std','mesh_90_percentiles','totalprecip']
input_ds = ds.copy()
for item in to_drop:
    if item in ds:
        input_ds = input_ds.drop(item)
        
#make the array of all the hrrr variable fields
variables = []
for variable in input_ds:
    variables.append(ds[variable])
input_array = np.stack(variables)

ds.close()

#Format the ml input and output arrays so the unet reads them correctly
input_array = np.swapaxes(input_array,0,1)
input_array = np.swapaxes(input_array,1,3)
output_array = np.swapaxes(output_array,1,2)


#convert output_array to categorical data
output_array = keras.utils.to_categorical(output_array, num_classes=4)


#Save out the metadata
rows = []
with open(metadata_outfile_name, 'a') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['Time'] + list(time))
    csvwriter.writerow(['Lat'] + list(lat))
    csvwriter.writerow(['Lon'] + list(lon))
    csvwriter.writerow(['MESH Fraction'] + list(meshfrac))
    csvwriter.writerow(['mesh_90_mean'] + list(mesh_90_mean))
    csvwriter.writerow(['mesh_90_std'] + list(mesh_90_std))
    csvwriter.writerow(['mesh_90_percentiles_min'] + list(mesh_90_percentiles[:,:,0]))
    csvwriter.writerow(['mesh_90_percentiles_10'] + list(mesh_90_percentiles[:,:,1]))
    csvwriter.writerow(['mesh_90_percentiles_25'] + list(mesh_90_percentiles[:,:,2]))
    csvwriter.writerow(['mesh_90_percentiles_50'] + list(mesh_90_percentiles[:,:,3]))
    csvwriter.writerow(['mesh_90_percentiles_75'] + list(mesh_90_percentiles[:,:,4]))
    csvwriter.writerow(['mesh_90_percentiles_90'] + list(mesh_90_percentiles[:,:,5]))
    csvwriter.writerow(['mesh_90_percentiles_max'] + list(mesh_90_percentiles[:,:,6]))

    
#read in the unet
model = keras.models.load_model(checkpoint_path)
    
#evaluate the unet on the testing data
y_hat = model.predict(input_array)

#make a dataset of the true and predicted patch data
ds_return = xr.Dataset(data_vars=dict(true_no_hail = (["patch", "x", "y"], output_array[:,:,:,0]),
                            true_hail = (["patch", "x", "y"], output_array[:,:,:,1]),
                            true_sev_hail = (["patch", "x", "y"], output_array[:,:,:,2]),
                            true_sig_sev_hail = (["patch", "x", "y"], output_array[:,:,:,3]),
                            predicted_no_hail = (["patch", "x", "y"], y_hat[:,:,:,0]),
                            predicted_hail = (["patch", "x", "y"], y_hat[:,:,:,1]),
                            predicted_sev_hail = (["patch", "x", "y"], y_hat[:,:,:,2]),
                            predicted_sig_sev_hail = (["patch", "x", "y"], y_hat[:,:,:,3])),
                    coords=dict(patch = range(y_hat.shape[0]),
                            x = range(32),
                            y = range(32)))

#save out the prediction and truth values
ds_return.to_netcdf(predictions_outfile_name)