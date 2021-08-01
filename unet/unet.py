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
metadata_outfile_name = checkpoint_path + 'metadata.csv'
#where to save training history?
model_history_name = checkpoint_path + 'history.csv'

#define hyperparameters
batch_size = 32
num_epochs = 2000
num_up_down_layers = 2
filter_num = [2,4,8,16,32]
learning_rate = 1e-4
loss_function = keras.losses.CategoricalCrossentropy() #losses.tversky #


#read in the data
infiles = glob.glob(path_to_patches)
#open all the data in one DataArray
ds = xr.open_mfdataset(infiles, concat_dim='n_samples',combine='nested', parallel=True, engine='netcdf4')

#isolate only the testing data
train = ds.where(ds.time <= np.datetime64('2019-07-01'),drop=True)#.dropna(dim='n_samples')

#Separate the data array into ml input, ml output and metadata
#output
output_ds = train.mesh
output_array = output_ds.values

#metadata
meshfrac = train.meshfrac.values
time = train.time.values
lon = train.lon.values
lat = train.lat.values
mesh_90_mean = train.mesh_90_mean.values
mesh_90_std = train.mesh_90_std.values
mesh_90_percentiles = train.mesh_90_percentiles.values

#input
to_drop = ['mesh','mesh_90','meshfrac','time','lon','lat','mesh_90_mean','mesh_90_std','mesh_90_percentiles','totalprecip']
input_ds = train.copy()
for item in to_drop:
    if item in train:
        input_ds = input_ds.drop(item)

#make the input arrays in the correct format for training
variables = []
for variable in input_ds:
    print(variable)
    print(input_ds[variable].values.shape)
    variables.append(input_ds[variable])
input_array = np.stack(variables)

#Format the ml input and output arrays so the unet reads them correctly
input_array = np.swapaxes(input_array,0,1)
input_array = np.swapaxes(input_array,1,3)
output_array = np.swapaxes(output_array,1,2)

#convert output_array to categorical data
output_array = keras.utils.to_categorical(output_array, num_classes=4)


#define some of the hyperparameters based on the input data
patch_size = input_array.shape[1]
num_inputs = input_array.shape[3]

#Save out the metadata
fields = ['Time', 'Lat', 'Lon', 'Rel_Freq_MESH', 'Mesh_90_Mean', 'Mesh_90_std', 'Mesh_90_Percentiles'] 
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
    
#make a basic unet
model = models.unet_2d((patch_size,patch_size,num_inputs), filter_num = filter_num, n_labels = 4, 
                        stack_num_down = num_up_down_layers, stack_num_up = num_up_down_layers, 
                        activation = 'LeakyReLU', output_activation = 'Softmax', 
                        batch_norm = True, pool = True, unpool = True, name = 'multi_class_unet')


#compile the unet
adam = Adam(learning_rate=learning_rate)
model.compile(loss=loss_function, optimizer=adam, metrics=tf.keras.metrics.Accuracy())

#define how the model will be trained and how training history recorded
early_stopping = EarlyStopping('loss',patience=500,verbose=1)
checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                        monitor='loss', verbose=1, save_best_only=True, 
                        save_weights_only=False, save_freq="epoch")
            
#print out the model architecture            
model.summary()

#train the model
trained_model = model.fit(input_array, output_array, epochs=num_epochs, batch_size=32, validation_freq=10,
                        callbacks=[early_stopping, checkpoint], verbose=2, workers=2, 
                        use_multiprocessing=True, max_queue_size=100)

#save out the loss, accuracy
rows = []
for key in trained_model.history.keys():
    rows.append([key] + list(trained_model.history[key]))
with open(model_history_name, 'a') as csvfile:
    csvwriter = csv.writer(csvfile) 
    csvwriter.writerows(rows)