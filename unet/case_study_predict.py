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


#path to the model
checkpoint_path = 'unet_data/final_cce_2000_1en4'

#where to save file
predictions_outfile_name = 'case_study_prediction.nc'

#load in the datasets
hrrr = xr.open_dataset('/ourdisk/hpc/ai2es/hail/ml_input/20210429/20210429_0200.nc')
mrms = xr.open_dataset('/ourdisk/hpc/ai2es/hail/ml_output_2/20210429/20210429_0200.nc')

#isolate the correct patch
patch = hrrr.isel(x=slice(400,432), y=slice(890,922))
labels_patch = mrms.isel(x=slice(400,432), y=slice(890,922))
mesh_class_patch = mrms.mesh.isel(x=slice(400,432), y=slice(890,922))
mesh_90_patch = mrms.mesh_90.isel(x=slice(400,432), y=slice(890,922))

#add mesh and mesh_90 to the patch
patch["mesh_90"] = (("x","y"), mesh_90_patch)
patch["mesh"] = (("x","y"), mesh_class_patch)

#redefine
ds = patch

#make output array
output_ds = ds.mesh
output_array = output_ds.values

#make input array
to_drop = ['mesh','mesh_90','totalprecip','hail','ltng']
input_ds = ds.copy()
for item in to_drop:
    if item in ds:
        input_ds = input_ds.drop(item)
variables = []
for variable in input_ds:
    print(variable)
    print(ds[variable].values.shape)
    variables.append(ds[variable])
input_array = np.stack(variables)

#close the dataset
ds.close()


#Format the ml input and output arrays so the unet reads them correctly
test = np.zeros((1,32,32,16))
test[0] = np.swapaxes(input_array,0,2)
input_array = test
output_array = np.swapaxes(output_array,0,1)


#convert output_array to categorical data
output_array = keras.utils.to_categorical(output_array, num_classes=4)


#make a basic unet
model = keras.models.load_model(checkpoint_path)

#run the unet on the data
y_hat = model.predict(input_array)

#make a dataarray of the new predictions
ds_return = xr.Dataset(data_vars=dict(true_no_hail = (["x", "y"], output_array[:,:,0]),
                            true_hail = (["x", "y"], output_array[:,:,1]),
                            true_sev_hail = (["x", "y"], output_array[:,:,2]),
                            true_sig_sev_hail = (["x", "y"], output_array[:,:,3]),
                            predicted_no_hail = (["x", "y"], y_hat[0,:,:,0]),
                            predicted_hail = (["x", "y"], y_hat[0,:,:,1]),
                            predicted_sev_hail = (["x", "y"], y_hat[0,:,:,2]),
                            predicted_sig_sev_hail = (["x", "y"], y_hat[0,:,:,3])),
                    coords=dict(x = range(32),
                            y = range(32)))

#save out the predictions
ds_return.to_netcdf(predictions_outfile_name)