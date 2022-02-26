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
import matplotlib.pyplot as plt



checkpoint_path = 'unet_test_data/regression_test19/'
predictions_outfile_name = 'case_study_regression_prediction4.nc'

infiles = (

patch = hrrr.isel(x=slice(200,800), y=slice(600,1200))
labels_patch = mrms.isel(x=slice(200,800), y=slice(600,1200))
mesh_class_patch = mrms.mesh.isel(x=slice(200,800), y=slice(600,1200))
mesh_90_patch = mrms.mesh_90.isel(x=slice(200,800), y=slice(600,1200))

patch["mesh_90"] = (("x","y"), mesh_90_patch)
patch["mesh"] = (("x","y"), mesh_class_patch)



ds = patch

output_ds = ds.mesh_90
output_array = output_ds.values

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

#Input in form: vars, y, x We want it in  (x, y, variables)
#Output in form 
#Format the ml input and output arrays so the unet reads them correctly
test = np.zeros((1,32,32,16))
test[0] = np.swapaxes(input_array,0,2)
input_array = test
print(input_array.shape)
#print(input_array)
#input_array = np.swapaxes(input_array,1,3)
output_array = np.swapaxes(output_array,0,1)#,2)
print(output_array.shape)

#convert output_array to categorical data
#output_array = keras.utils.to_categorical(output_array, num_classes=4)


#make a basic unet
model = keras.models.load_model(checkpoint_path)


y_hat = model.predict(input_array)

print(y_hat.shape)


ds_return = xr.Dataset(data_vars=dict(true_hail = (["x", "y"], output_array),
                            predicted_hail = (["x", "y"], y_hat[0,:,:,0])),
                         coords=dict(x = range(600),      
                                y = range(600)))
                
ds_return.to_netcdf(predictions_outfile_name)

#def plot_ml_output(output, idx = 0):
 # keys = ["Regression Output"]
  #fig, axes = plt.subplots(1,2,figsize=(20,5))
  #axes = axes.ravel()
  #or i,ax in enumerate(axes[:-1]):
    #pm = ax.imshow(output[:,:,i].T,cmap = 'cividis',vmin = 0,vmax = 50)
    #ax.set_title(keys[i])
    #plt.colorbar(pm, ax=ax,shrink=0.60)
  #axes[-1].axis(False)
  #plt.tight_layout()
  
#plot_ml_output(y_hat)
#plt.savefig('case_study.png',dpi=300)

