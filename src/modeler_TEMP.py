import xarray as xr
import numpy as np
import glob
from keras_unet_collection import models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from tensorflow import keras
import py3nvml


def predict():
    label_patches_dir = "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/REU_run/patches/test_norman_apr_28_casestudy/mrms_labels/*"
    feature_patches_dir = "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/REU_run/patches/test_norman_apr_28_casestudy/hrrr_examples/*"
    checkpoint_path = "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/REU_run/model_data/run_1"
    predictions_outfile_name = "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/REU_run/patches/predicted_norman_apr_28_casestudy/predictions_norm.nc"

    #output
    label_files = glob.glob(label_patches_dir)
    label_files.sort()
    #open all the data in one DataArray
    labels_ds = xr.open_mfdataset(label_files, concat_dim='n_samples',combine='nested', parallel=True, engine='netcdf4', coords="all")
    output_ds = labels_ds.mesh
    output_array = output_ds.values
    lats = labels_ds["latitude"].to_numpy()
    lons = labels_ds["longitude"].to_numpy()
    output_ds.close()

    #input
    feature_files = glob.glob(feature_patches_dir)
    feature_files.sort()
    #open all the data in one DataArray
    feature_ds = xr.open_mfdataset(feature_files, concat_dim='n_samples',combine='nested', parallel=True, engine='netcdf4', coords="all")
    input_array = feature_ds.to_array().to_numpy()
    feature_ds.close()

    # Format the ml input and output arrays so the unet reads them correctly
    # unet_2d needs [example_n, img_row, img_col, feature_num]
    input_array = np.swapaxes(input_array,0,1)
    input_array = np.swapaxes(input_array,1,3)
    output_array = np.swapaxes(output_array,1,2)

    #convert output_array to categorical data
    # TODO: MAKE SURE THIS ISNT HARD CODED
    output_array_cat = keras.utils.to_categorical(output_array, num_classes=4)

    #read in the unet
    model = keras.models.load_model(checkpoint_path)
        
    #evaluate the unet on the testing data
    y_hat = model.predict(input_array)

    y_hat_full = np.argmax(y_hat,axis=3)

    #make a dataset of the true and predicted patch data
    # TODO: Avoid hardcoding here
    ds_return = xr.Dataset(data_vars=dict(true_full_data = (["patch", "x", "y"], output_array),
                                true_no_hail = (["patch", "x", "y"], output_array_cat[:,:,:,0]),
                                true_hail = (["patch", "x", "y"], output_array_cat[:,:,:,1]),
                                true_sev_hail = (["patch", "x", "y"], output_array_cat[:,:,:,2]),
                                true_sig_sev_hail = (["patch", "x", "y"], output_array_cat[:,:,:,3]),
                                predicted_no_hail = (["patch", "x", "y"], y_hat[:,:,:,0]),
                                predicted_hail = (["patch", "x", "y"], y_hat[:,:,:,1]),
                                predicted_sev_hail = (["patch", "x", "y"], y_hat[:,:,:,2]),
                                predicted_sig_sev_hail = (["patch", "x", "y"], y_hat[:,:,:,3]),
                                predicted_full_data = (["patch", "x", "y"], y_hat_full)),
                        coords=dict(x = range(32),
                                y = range(32),
                                latitude = (["patch", "x", "y"], lats),
                                longitude = (["patch", "x", "y"], lons),
                                patch = range(y_hat.shape[0])))

    #save out the prediction and truth values
    ds_return.to_netcdf(predictions_outfile_name)




def train():
    label_patches_dir = "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/REU_run/patches/train_val/mrms_labels/*"
    feature_patches_dir = "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/REU_run/patches/train_val/hrrr_examples/*"
    checkpoint_path = "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/REU_run/model_data/run_2"

    # define hyperparameters
    batch_size = 32
    num_epochs = 2000
    num_up_down_layers = 2
    filter_num = [2,4,8,16,32]
    learning_rate = 1e-4
    loss_function = keras.losses.CategoricalCrossentropy() #losses.tversky #


    #output
    label_files = glob.glob(label_patches_dir)
    label_files.sort()
    #open all the data in one DataArray
    labels_ds = xr.open_mfdataset(label_files, concat_dim='n_samples',combine='nested', parallel=True, engine='netcdf4')
    output_ds = labels_ds.mesh
    output_array = output_ds.values
    output_ds.close()

    #input
    feature_files = glob.glob(feature_patches_dir)
    feature_files.sort()
    #open all the data in one DataArray
    feature_ds = xr.open_mfdataset(feature_files, concat_dim='n_samples',combine='nested', parallel=True, engine='netcdf4')
    input_array = feature_ds.to_array().to_numpy()
    feature_ds.close()

    # Format the ml input and output arrays so the unet reads them correctly
    # unet_2d needs [example_n, img_row, img_col, feature_num]
    input_array = np.swapaxes(input_array,0,1)
    input_array = np.swapaxes(input_array,1,3)
    output_array = np.swapaxes(output_array,1,2)

    #convert output_array to categorical data
    # TODO: MAKE SURE THIS ISNT HARD CODED
    output_array = keras.utils.to_categorical(output_array, num_classes=4)

    #define some of the hyperparameters based on the input/output data
    patch_size = input_array.shape[1]
    num_inputs = input_array.shape[3]
    n_labels = output_array.shape[-1]

    #make a basic unet
    # TODO: MAKE SURE THIS ISNT HARD CODED
    model = models.unet_2d((patch_size,patch_size,num_inputs), filter_num = filter_num, n_labels = n_labels, 
                            stack_num_down = num_up_down_layers, stack_num_up = num_up_down_layers, 
                            activation = 'LeakyReLU', output_activation = 'Softmax', 
                            batch_norm = True, pool = True, unpool = True, name = 'multi_class_unet')


    #compile the unet
    adam = Adam(learning_rate=learning_rate)
    model.compile(loss=loss_function, optimizer=adam, metrics=tf.keras.metrics.Accuracy())

    #define how the model will be trained and how training history recorded
    # TODO: Switch to val loss?
    early_stopping = EarlyStopping(monitor='loss',patience=500,verbose=1)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                            monitor='loss', verbose=1, save_best_only=True, 
                            save_weights_only=False, save_freq="epoch")
                
    #print out the model architecture            
    model.summary()

    # THIS IS TEMP
    n_examples = input_array.shape[0]
    val_fraction = int(n_examples*0.9)
    input_array_val = input_array[val_fraction:,:,:,:]
    output_array_val = output_array[val_fraction:,:,:,:]
    input_array = input_array[:val_fraction,:,:,:]
    output_array = output_array[:val_fraction,:,:,:]

    #train the model
    trained_model = model.fit(input_array, output_array, epochs=num_epochs, batch_size=batch_size, validation_freq=10,
                            validation_data=(input_array_val, output_array_val), callbacks=[early_stopping, checkpoint], verbose=1, workers=2, 
                            use_multiprocessing=True, max_queue_size=100)



if __name__ == "__main__":
    #number of GPUS I need:
    n_gpu = 1

    #find out how many are compltely free
    free_gpus = py3nvml.get_free_gpus()

    #count how many are free
    avail_gpu_ids = np.where(free_gpus)[0]

    #if there arent enough print it out
    if len(avail_gpu_ids) < n_gpu:
        print('Not enough GPUs, your job might fail')
    else:
        #if there are enough, the select the ones you need
        py3nvml.grab_gpus(num_gpus=n_gpu, gpu_select=[0])
    
    tf.config.threading.set_intra_op_parallelism_threads(32)
    tf.config.threading.set_inter_op_parallelism_threads(32)

    train()
    # predict()