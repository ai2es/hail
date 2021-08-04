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

meshfrac = ds.meshfrac.values
time = ds.time.values
lon = ds.lon.values
lat = ds.lat.values


input_ds = ds.drop(('mesh','mesh_90','meshfrac','time','lon','lat'))
variables = []
for variable in input_ds:
    variables.append(ds[variable])
    print(variable)
input_array = np.stack(variables)


 

#Format the ml input and output arrays so the unet reads them correctly
input_array = np.swapaxes(input_array,0,1)
input_array = np.swapaxes(input_array,1,3)
output_array = np.swapaxes(output_array,1,2)

#convert output_array to categorical data
#output_array = to_categorical(output_array, num_classes=4)


#define some of the input parameters to the ml files (instead of hardcoding them)
num_up_down_layers = 4
patch_size = input_array.shape[1]
batch_size = input_array.shape[0]
num_inputs = input_array.shape[3]
#print(patch_size)
#print(batch_size)
#print(num_inputs)
#input_array = input_array.swapaxes(0,3)

filter_num = []
size = patch_size
for i in range(num_up_down_layers + 1):
    filter_num = [int(size/(2**i))] + filter_num
#print(filter_num)


#Save out the metadata
fields = ['Time', 'Lat', 'Lon', 'Rel_Freq_MESH'] 

rows = []
for i in range(batch_size):
    rows.append([time[i][0],lat[i][0],lon[i][0],meshfrac[i][0]])

with open(metadata_outfile_name, 'a') as csvfile:
    csvwriter = csv.writer(csvfile) 
    csvwriter.writerow(fields) 
    csvwriter.writerows(rows)



#make a basic unet
model = models.unet_2d((patch_size,patch_size,num_inputs), filter_num = filter_num, n_labels = 1, 
                        stack_num_down = num_up_down_layers, stack_num_up = num_up_down_layers, 
                        activation = 'LeakyReLU', output_activation = 'ReLU', 
                        batch_norm = True, pool = True, unpool = True, name = 'regression_unet')

#compile the unet
learning_rate = 1e-4
loss_function = keras.losses.MeanSquaredError()

adam = Adam(learning_rate=learning_rate)
model.compile(loss=loss_function, optimizer=adam, metrics=tf.keras.metrics.Accuracy())

#train the model

early_stopping = EarlyStopping('loss',patience=500,verbose=2)
checkpoint = tf.keras.callbacks.ModelCheckpoint('unet_test_data/regression_test_2',
                        monitor='loss', verbose=1, save_best_only=True, 
                        save_weights_only=False, save_freq='epoch')

trained_model = model.fit(input_array, output_array, epochs=250, batch_size=batch_size, validation_freq=10,
                        callbacks=[early_stopping, checkpoint], verbose=2, workers=2, 
                        use_multiprocessing=True, max_queue_size=100)
                        
acc = trained_model.history['accuracy']
#val_acc = trained_model.history['val_accuracy']

loss = trained_model.history['loss']
#val_loss = trained_model.history['val_loss']

epochs_range = range(250)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
#plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
#plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

#validation_freq=5,

#steps_per_epoch=13,
#validation_steps=5,



#####################OLD CODE BELOW#####################
#read in the data:
#ds = xr.open_dataset(infile)

#input_ds = xr.open_dataset("unet_test_data/HRRR_allinput_0000_2200.nc")
#output_ds = xr.open_dataset("unet_test_data/HRRR_hail_0100_2300.nc")

#put the data into an array
#input_array = np.stack([input_ds.hail.values,input_ds.refc.values,input_ds.cape.values])

#crop the arrays to be divisible by 2, split into testing, training, validation
#X_array = input_array[:,:,500:756,1022:1278]
#Y_array = output_ds.hail[:,500:756,1022:1278].values

#swap the axes to fit ML model input
#X_array = np.swapaxes(X_array,0,1)
#X_array = np.swapaxes(X_array,1,3)
#Y_array = np.swapaxes(Y_array,1,2)