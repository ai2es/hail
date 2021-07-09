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

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

#make a basic unet
model = models.unet_2d((256,256,3), filter_num = [64,128,256], n_labels = 1, 
                        stack_num_down = 2, stack_num_up = 2, 
                        activation = 'LeakyReLU', output_activation = 'ReLU', 
                        batch_norm = True, pool = True, unpool = True, name = 'unet')

#compile the unet
learning_rate = 1e-4
loss_function = keras.losses.MeanSquaredError()

adam = Adam(learning_rate=learning_rate)
model.compile(loss=loss_function, optimizer=adam, metrics=tf.keras.metrics.Accuracy())

#train the model

early_stopping = EarlyStopping('loss',patience=500,verbose=2)
checkpoint = tf.keras.callbacks.ModelCheckpoint('unet_test_data/model_test',
                        monitor='loss', verbose=1, save_best_only=True, 
                        save_weights_only=False, save_freq='epoch')

trained_model = model.fit(X_array, Y_array, epochs=100, batch_size=23, validation_freq=10,
                        callbacks=[early_stopping, checkpoint], verbose=2, workers=2, 
                        use_multiprocessing=True, max_queue_size=100)
#validation_freq=5,

#steps_per_epoch=13,
#validation_steps=5,


#Done?



