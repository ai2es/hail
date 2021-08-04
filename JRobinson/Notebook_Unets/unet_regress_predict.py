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


path_to_patches = '/ourdisk/hpc/ai2es/hail/images_for_learning/*'
checkpoint_path = 'unet_test_data/regression_test_8/'
metadata_outfile_name = checkpoint_path + 'metadata_predict.csv'
predictions_outfile_name = checkpoint_path + 'predictions.nc'

#read in the data
infiles = glob.glob(path_to_patches)
ds = xr.open_mfdataset(infiles, concat_dim='n_samples',combine='nested', parallel=True, engine='netcdf4')


#Separate the data array into ml input, ml output and metadata
output_ds = ds.mesh
output_array = output_ds.values

meshfrac = ds.meshfrac.values
time = ds.time.values
lon = ds.lon.values
lat = ds.lat.values

mesh_90_mean = ds.mesh_90_mean.values
mesh_90_std = ds.mesh_90_std.values
mesh_90_percentiles = ds.mesh_90_percentiles.values


input_ds = ds.drop(('mesh','mesh_90','meshfrac','time','lon','lat','mesh_90_mean','mesh_90_std','mesh_90_percentiles'))
variables = []
for variable in input_ds:
    print(variable)
    print(ds[variable].values.shape)
    variables.append(ds[variable])
input_array = np.stack(variables)

#Format the ml input and output arrays so the unet reads them correctly
input_array = np.swapaxes(input_array,0,1)
input_array = np.swapaxes(input_array,1,3)
output_array = np.swapaxes(output_array,1,2)


#convert output_array to categorical data
output_array = keras.utils.to_categorical(output_array, num_classes=4)


#Normalize the Inputs
mu = np.mean(input_array,axis=(1,2,3))
sigma = np.std(input_array,axis=(1,2,3))
#need to reshape mu and sigma to match shape 
mu = np.tile(mu[:,np.newaxis],(1,input_array.shape[1]))
mu = np.tile(mu[:,:,np.newaxis],(1,1,input_array.shape[2]))
mu = np.tile(mu[:,:,:,np.newaxis],(1,1,1,input_array.shape[3]))
sigma = np.tile(sigma[:,np.newaxis],(1,input_array.shape[1]))
sigma = np.tile(sigma[:,:,np.newaxis],(1,1,input_array.shape[2]))
sigma = np.tile(sigma[:,:,:,np.newaxis],(1,1,1,input_array.shape[3]))
input_array_scaled = (input_array - mu)/sigma


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
    csvwriter.writerow(['mesh_90_percentiles_90'] + list(mesh_90_percentiles[:,:,1]))
    csvwriter.writerow(['mesh_90_percentiles_max'] + list(mesh_90_percentiles[:,:,2]))

    
#make a basic unet
model = keras.models.load_model(checkpoint_path)

y_hat = model.predict(input_array)


ds_return = xr.Dataset(data_vars=dict(true_no_hail = (["patch", "x", "y"], output_array[:,:,0]),
                            true_hail = (["patch", "x", "y"], output_array[:,:,1]),
                            predicted_hail = (["patch", "x", "y"], y_hat[:,:,:,0]),
                    coords=dict(patch = range(y_hat.shape[0]),
                            x = range(32),
                            y = range(32))))
                
ds_return.to_netcdf(predictions_outfile_name)