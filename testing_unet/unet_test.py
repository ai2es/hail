from keras_unet_collection import models, losses
import tensorflow as tf
from tensorflow import keras
import xarray as xr
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

#read in the data:
input_ds = xr.open_dataset("unet_test_data/HRRR_allinput_0000_2200.nc")
output_ds = xr.open_dataset("unet_test_data/HRRR_hail_0100_2300.nc")

#put the data into an array
input_array = np.stack([input_ds.hail.values,input_ds.refc.values,input_ds.cape.values])

#crop the arrays to be divisible by 2, split into testing, training, validation
X_array = input_array[:,:,500:756,1022:1278]
Y_array = output_ds.hail[:,500:756,1022:1278].values

#swap the axes to fit ML model input
X_array = np.swapaxes(X_array,0,1)
X_array = np.swapaxes(X_array,1,3)
Y_array = np.swapaxes(Y_array,1,2)

#put a log scale on the outputs because of rarity of events
#Y_array = np.log10(Y_array)


model = keras.models.load_model('unet_test_data/model_test/')

output = model.predict(X_array)

#plt.pcolormesh(output[20,:,:,0])
#plt.savefig("unet_test_100ep.png")