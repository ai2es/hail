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


print('Hello World')
infiles = glob.glob('/ourdisk/hpc/ai2es/hail/images_for_learning/*')
metadata_outfile_name = 'unet_regression_test7_output.csv'


#read in the data
ds = xr.open_mfdataset(infiles, concat_dim='n_samples',combine='nested', parallel=True)
print(ds)

#Separate the data array into ml input, ml output and metadata
output_ds = ds.mesh_90
output_array = output_ds.values


mesh_90_percentiles = ds.mesh_90_percentiles.values
mesh_90_mean = ds.mesh_90_mean.values
mesh_90_std = ds.mesh_90_std.values
time = ds.time.values
lon = ds.lon.values
lat = ds.lat.values

to_drop = ['mesh','mesh_90','meshfrac','time','lon','lat','mesh_90_mean','mesh_90_std','mesh_90_percentiles','totalprecip']
for item in to_drop:
    if item in ds:
        input_ds = ds.drop(item)
        
input_array = input_ds.values

input_array