import xarray as xr
import numpy as np
import glob
from keras_unet_collection import models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from tensorflow import keras
import py3nvml
import argparse
import json
from sklearn.model_selection import KFold
import os


def predict(args):
    label_patches_dir = args["labels"]
    feature_patches_dir = args["examples"]
    checkpoint_path = args["single_checkpoint"]
    predictions_outfile = args["predictions_outfile"]

    # Define all settings
    label_vars_to_drop = args["label_vars_to_drop"]
    feature_vars_to_drop = args["feature_vars_to_drop"]

    #output
    label_files = glob.glob(label_patches_dir)
    label_files.sort()
    #open all the data in one DataArray
    labels_ds = xr.open_mfdataset(label_files, concat_dim='n_samples',combine='nested', parallel=True, engine='netcdf4', coords="all")
    lats = labels_ds["lat"].to_numpy()
    lons = labels_ds["lon"].to_numpy()

    #input
    feature_files = glob.glob(feature_patches_dir)
    feature_files.sort()
    #open all the data in one DataArray
    features_ds = xr.open_mfdataset(feature_files, concat_dim='n_samples',combine='nested', parallel=True, engine='netcdf4', coords="all")

    labels_ds = labels_ds.drop(label_vars_to_drop)
    features_ds = features_ds.drop(feature_vars_to_drop)

    # Convert to xarray arrays. (Still maintains xarray's smart memory management)
    output_array = labels_ds.to_array()
    labels_ds.close()
    input_array = features_ds.to_array()
    features_ds.close()

    input_array = input_array.transpose("n_samples", ...)
    input_array = input_array.transpose(..., "variable")

    output_array = output_array.transpose("n_samples", ...)
    output_array = output_array.transpose(..., "variable")

    output_array_np = output_array.to_numpy()
    input_array_np = input_array.to_numpy()

    #read in the unet
    model = keras.models.load_model(checkpoint_path)
        
    #evaluate the unet on the testing data
    y_hat = model.predict(input_array_np)

    #make a dataset of the true and predicted patch data
    # TODO: Avoid hardcoding here
    ds_return = xr.Dataset(data_vars=dict(true_no_hail = (["patch", "x", "y"], output_array_np[:,:,:,0]),
                                true_hail = (["patch", "x", "y"], output_array_np[:,:,:,1]),
                                # true_sev_hail = (["patch", "x", "y"], output_array[:,:,:,2]),
                                # true_sig_sev_hail = (["patch", "x", "y"], output_array[:,:,:,3]),
                                predicted_no_hail = (["patch", "x", "y"], y_hat[:,:,:,0]),
                                predicted_hail = (["patch", "x", "y"], y_hat[:,:,:,1]),
                                # predicted_sev_hail = (["patch", "x", "y"], y_hat[:,:,:,2]),
                                # predicted_sig_sev_hail = (["patch", "x", "y"], y_hat[:,:,:,3])
                                ),
                        coords=dict(x = range(output_array_np.shape[2]),
                                y = range(output_array_np.shape[1]),
                                latitude = (["patch", "x", "y"], lats),
                                longitude = (["patch", "x", "y"], lons),
                                patch = range(y_hat.shape[0])))

    #save out the prediction and truth values
    ds_return.to_netcdf(predictions_outfile)


def train(args):
    # Get paths
    label_patches_dir = args["labels"]
    feature_patches_dir = args["examples"]
    checkpoint_path_root = args["model_checkpoints"]

    # define all settings
    batch_size = args["batch_size"]
    val_freq = args["val_freq"]
    steps_per_epoch = args["steps_per_epoch"]
    num_epochs = args["epochs"]
    num_up_down_layers = args["num_up_down_layers"]
    filter_num = args["filter_nums"]
    learning_rate = args["learning_rate"]
    loss_function = args["loss_func"]
    training_metrics = args["metrics"]
    main_activations = args["main_activations"]
    output_activation = args["output_activation"]
    label_vars_to_drop = args["label_vars_to_drop"]
    feature_vars_to_drop = args["feature_vars_to_drop"]
    verbose = args["verbose"]
    patience = args["patience"]
    min_delta = args["min_delta"]
    n_folds = args["n_folds"]

    #output
    label_files = glob.glob(label_patches_dir)
    label_files.sort()
    #open all the data in one xarray dataset
    # dataset.chunk was tried as well but it was found to be unstable with tensorflow when trying to load every variable in a .nc file into each chunk
    labels_ds = xr.open_mfdataset(label_files, concat_dim='n_samples', combine='nested', parallel=True, engine='netcdf4', chunks={"n_samples": 500}) # Can only chunk maximum of the size of one file for concat_dim (1000)

    #input
    feature_files = glob.glob(feature_patches_dir)
    feature_files.sort()
    #open all the data in one xarray dataset
    # dataset.chunk was tried as well but it was found to be unstable with tensorflow when trying to load every variable in a .nc file into each chunk
    features_ds = xr.open_mfdataset(feature_files, concat_dim='n_samples', combine='nested', parallel=True, engine='netcdf4', chunks={"n_samples": 500}) # Can only chunk maximum of the size of one file for concat_dim (1000)

    # labels_ds = labels_ds.chunk(chunks={"n_samples": 3000, "lat_dim": 64, "lon_dim": 64})
    # features_ds = features_ds.chunk(chunks={"n_samples": 3000, "lat_dim": 64, "lon_dim": 64})

    labels_ds = labels_ds.drop(label_vars_to_drop)
    features_ds = features_ds.drop(feature_vars_to_drop)

    # Convert to xarray arrays. (Still maintains xarray's smart memory management)
    output_array = labels_ds.to_array()
    labels_ds.close()
    input_array = features_ds.to_array()
    features_ds.close()

    ####################### HOW THIS USED TO BE DONE ##########################
    # Format the ml input and output arrays so the unet reads them correctly
    # unet_2d needs [example_n, img_row, img_col, feature_num]
    # input_array = np.swapaxes(input_array,0,1)
    # input_array = np.swapaxes(input_array,1,3)
    # output_array = np.swapaxes(output_array,1,2)
    ############################################################################

    input_array = input_array.transpose("n_samples", ...)
    input_array = input_array.transpose(..., "variable")

    output_array = output_array.transpose("n_samples", ...)
    output_array = output_array.transpose(..., "variable")

    # output_array = output_array[{"variable": 0}]

    #convert output_array to categorical data
    # output_array = keras.utils.to_categorical(output_array, num_classes=num_classes)

    # OLD WAY TO FOLD DATA
    # n_examples = input_array.shape[0]
    # val_fraction = int(n_examples*0.9)
    # input_array_val = input_array[val_fraction:,:,:,:]
    # output_array_val = output_array[val_fraction:,:,:,:]
    # input_array = input_array[:val_fraction,:,:,:]
    # output_array = output_array[:val_fraction,:,:,:]

    #define some of the hyperparameters based on the input/output data
    patch_size = input_array.shape[1]
    num_inputs = input_array.shape[3]
    n_labels = output_array.shape[-1]

    for i, (train_index,val_index) in enumerate(KFold(n_folds).split(input_array, output_array)):
        checkpoint_path = os.path.join(checkpoint_path_root, "fold_" + str(i) + "/")

        if not os.path.exists(checkpoint_path):
            os.mkdir(checkpoint_path)

        input_train, input_val_dataarray = input_array[train_index], input_array[val_index]
        output_train, output_val_dataarray  = output_array[train_index], output_array[val_index]

        input_val = input_val_dataarray.to_numpy()
        input_val_dataarray.close()
        output_val = output_val_dataarray.to_numpy()
        output_val_dataarray.close()

        # output_val = output_val.chunk(chunks={"n_samples": 39000})
        # input_val = input_val.chunk(chunks={"n_samples": 39000})

        #make a basic unet
        model = models.unet_2d((patch_size,patch_size,num_inputs), filter_num = filter_num, n_labels = n_labels, 
                                stack_num_down = num_up_down_layers, stack_num_up = num_up_down_layers, 
                                activation = main_activations, output_activation = output_activation, 
                                batch_norm = True, pool = True, unpool = True, name = 'unet')

        #compile the unet
        opt = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999,
                                    epsilon=None, decay=0.0, amsgrad=False)
        model.compile(loss=loss_function, optimizer=opt, metrics=training_metrics)

        #define how the model will be trained and how training history recorded
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, verbose=verbose, restore_best_weights=True, min_delta=min_delta)
        checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                monitor='val_loss', verbose=verbose, save_best_only=True, 
                                save_weights_only=False, save_freq="epoch")
                    
        #print out the model architecture            
        model.summary()

        #train the model
        # TODO: ensure validation_freq, batch_size, steps_per_epoch are all in settings.
        trained_model = model.fit(input_train, output_train, epochs=num_epochs, validation_freq=val_freq, batch_size=batch_size, steps_per_epoch=steps_per_epoch, shuffle=False, 
                                validation_data=(input_val, output_val), callbacks=[early_stopping, checkpoint], verbose=verbose)

        # TODO: Add pickle file for saved history here


def create_parser():
    '''
    Create argument parser
    '''
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description='Unet Model', fromfile_prefix_chars='@')

    # Predict only settings
    parser.add_argument('--single_checkpoint', type=str, default='/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/2022_07_26_run/saved_models/fold_0')
    parser.add_argument('--predictions_outfile', type=str, default='/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/2022_07_26_run/saved_models/fold_0')

    # Directories
    parser.add_argument('--args_json', type=str, default='/home/tgschmidt/hail/configs/config.json')
    parser.add_argument('--examples', type=str, default='/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/no_refl_trained_at_init_time_2022_08_03/patches/train_val/examples/*')
    parser.add_argument('--labels', type=str, default='/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/no_refl_trained_at_init_time_2022_08_03/patches/train_val/labels/*')
    parser.add_argument('--model_checkpoints', type=str, default='/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/no_refl_trained_at_init_time_2022_08_03/saved_models')

    # High-level commands
    parser.add_argument('--train', '-t', action='store_true')
    parser.add_argument('--verbose', '-v', action='count', default=1, help="Verbosity level")
    parser.add_argument('--n_gpus', type=int, default=1)
    parser.add_argument('--n_cpus', type=int, default=8)

    # Data
    parser.add_argument('--feature_vars_to_drop', type=str, nargs='+', default=['lon', 'lat'])
    parser.add_argument('--label_vars_to_drop', type=str, nargs='+', default=['time', 'lon', 'lat', 'MESH95'])

    # Training
    parser.add_argument('--epochs', type=int, default=2000, help='Training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=30, help="Patience for early termination")
    parser.add_argument('--min_delta', type=float, default=0.0, help="Minimum delta for early termination")
    parser.add_argument('--n_folds', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=50, help='Batch of patches to load at once')
    parser.add_argument('--val_freq', type=int, default=5)
    parser.add_argument('--steps_per_epoch', type=int, default=20)

    # Model configs
    parser.add_argument('--filter_nums', type=int, nargs='+', default=[4,8,16,32,64], help="Number of filters in each CNN")
    parser.add_argument('--num_up_down_layers', type=int, default=5)
    parser.add_argument('--loss_func', type=str, default="binary_crossentropy")
    parser.add_argument('--metrics', type=str, nargs='+', default=["binary_accuracy"])
    parser.add_argument('--main_activations', type=str, default="LeakyReLU")
    parser.add_argument('--output_activation', type=str, default="Sigmoid")
    
    return parser


if __name__ == "__main__":
    # Perform all the argument parsing
    parser = create_parser()
    args = parser.parse_args()
    with open(args.args_json) as json_file:
        json_dict = json.load(json_file)
    args = vars(args)
    args.update(json_dict)

    #number of GPUS and CPUS I need:
    n_gpu = args["n_gpus"]
    n_cpus = args["n_cpus"]

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
    
    tf.config.threading.set_intra_op_parallelism_threads(n_cpus)
    tf.config.threading.set_inter_op_parallelism_threads(n_cpus)

    if args["train"]:
        train(args)
    else:
        predict(args)