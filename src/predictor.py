import xarray as xr
import glob
from tensorflow import keras
import argparse
import json
# from custom_metrics import MaxCriticalSuccessIndex


def create_parser():
    '''
    Create argument parser
    '''
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description='Unet Model', fromfile_prefix_chars='@')

    # Predict only settings
    parser.add_argument('--single_checkpoint', type=str, default='/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/no_refl_trained_at_init_time_2022_08_03/saved_models/fold_0')
    parser.add_argument('--predictions_outfile', type=str, default='/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/no_refl_trained_at_init_time_2022_08_03/patches/predictions/y_hats.nc')
    # parser.add_argument('--feature_vars_to_drop', type=str, nargs='+', default=['lon', 'lat'])
    # parser.add_argument('--label_vars_to_drop', type=str, nargs='+', default=['time', 'lon', 'lat', 'MESH95'])

    # Data
    parser.add_argument('--args_json', type=str, default='/home/tgschmidt/hail/configs/predict.json')
    parser.add_argument('--examples', type=str, default='/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/no_refl_trained_at_init_time_2022_08_03/patches/test/examples/*')
    parser.add_argument('--labels', type=str, default='/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/no_refl_trained_at_init_time_2022_08_03/patches/test/labels/*')
    # parser.add_argument('--model_checkpoints', type=str, default='/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/no_refl_trained_at_init_time_2022_08_03/saved_models')
    # parser.add_argument('--tf_ds_dir', type=str, default='/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/trained_at_init_time_4_class_2022_08_09/patches/train_val/tf_datasets/*')

    # # High-level commands
    # parser.add_argument('--train', '-t', action='store_true')
    # parser.add_argument('--verbose', '-v', action='count', default=1, help="Verbosity level")
    # parser.add_argument('--n_gpus', type=int, default=1)
    # parser.add_argument('--n_cpus', type=int, default=12)

    # # Training
    # parser.add_argument('--epochs', type=int, default=2000, help='Training epochs')
    # parser.add_argument('--learning_rate', type=float, default=1e-4)
    # parser.add_argument('--patience', type=int, default=30, help="Patience for early termination")
    # parser.add_argument('--min_delta', type=float, default=0.0, help="Minimum delta for early termination")
    # parser.add_argument('--n_folds', type=int, default=10)
    # parser.add_argument('--batch_size', type=int, default=50, help='Batch of patches to load at once')
    # parser.add_argument('--val_freq', type=int, default=10)
    # parser.add_argument('--steps_per_epoch', type=int, default=20)

    # # Model configs
    # parser.add_argument('--filter_nums', type=int, nargs='+', default=[4,8,16,32,64], help="Number of filters in each CNN")
    # parser.add_argument('--num_up_down_layers', type=int, default=5)
    # parser.add_argument('--loss_func', type=str, default="binary_crossentropy")
    # parser.add_argument('--metrics', type=str, nargs='+', default=["binary_accuracy"])
    # parser.add_argument('--main_activations', type=str, default="LeakyReLU")
    # parser.add_argument('--output_activation', type=str, default="Sigmoid")
    
    return parser


if __name__ == "__main__":
    # Perform all the argument parsing
    parser = create_parser()
    args = parser.parse_args()
    with open(args.args_json) as json_file:
        json_dict = json.load(json_file)
    args = vars(args)
    args.update(json_dict)

    # #number of GPUS and CPUS I need:
    # n_gpu = args["n_gpus"]
    # n_cpus = args["n_cpus"]

    # #find out how many are compltely free
    # free_gpus = py3nvml.get_free_gpus()

    # #count how many are free
    # avail_gpu_ids = np.where(free_gpus)[0]

    # #if there arent enough print it out
    # if len(avail_gpu_ids) < n_gpu:
    #     raise Exception('Not enough GPUs to support your given "n_gpu" now. Maybe they are being used?')
    # else:
    #     #if there are enough, the select the ones you need
    #     py3nvml.grab_gpus(num_gpus=n_gpu, gpu_select=[0])

    # GPU check
    # physical_devices = tf.config.list_physical_devices('GPU') 
    # n_physical_devices = len(physical_devices)
    # if(n_physical_devices > 0):
    #     for device in physical_devices:
    #         tf.config.experimental.set_memory_growth(device, True)
    
    # tf.config.threading.set_intra_op_parallelism_threads(n_cpus)
    # tf.config.threading.set_inter_op_parallelism_threads(n_cpus)

    label_patches_dir = args["labels"]
    example_patches_dir = args["examples"]
    checkpoint_path = args["single_checkpoint"]
    predictions_outfile = args["predictions_outfile"]

    # Define all settings
    # label_vars_to_drop = args["label_vars_to_drop"]
    # feature_vars_to_drop = args["feature_vars_to_drop"]

    # glob files
    label_files = glob.glob(label_patches_dir)
    label_files.sort()
    example_files = glob.glob(example_patches_dir)
    example_files.sort()

    #open all the data in one xarray dataset
    example_ds_list = []
    label_ds_list = []
    for input_file, output_file in zip(example_files, label_files):
        example_ds_list.append(xr.open_dataset(input_file, decode_cf=False))
        label_ds_list.append(xr.open_dataset(output_file, decode_cf=False))

    examples_ds = xr.concat(example_ds_list, dim="n_samples")
    labels_ds = xr.concat(label_ds_list, dim="n_samples")

    label_keys = list(labels_ds.keys())
    label_dims = list(labels_ds.dims)

    # lats = examples_ds["lat"].to_numpy()
    # lons = examples_ds["lon"].to_numpy()
    # comp_dz = examples_ds["comp_dz"].to_numpy()

    # labels_ds = labels_ds.drop(label_vars_to_drop)
    # examples_ds = examples_ds.drop(feature_vars_to_drop)

    # Convert to xarray arrays. (Still maintains xarray's smart memory management)
    output_array = labels_ds.to_array()
    labels_ds.close()
    input_array = examples_ds.to_array()
    examples_ds.close()

    input_array = input_array.transpose("n_samples", ...)
    input_array = input_array.transpose(..., "variable")

    output_array = output_array.transpose("n_samples", ...)
    output_array = output_array.transpose(..., "variable")

    output_array_np = output_array.to_numpy()
    input_array_np = input_array.to_numpy()

    #read in the unet
    model = keras.models.load_model(checkpoint_path, compile=False)#, custom_objects={"max_csi": MaxCriticalSuccessIndex()})
        
    #evaluate the unet on the testing data
    y_hat = model.predict(input_array_np)

    new_ds_dict = {key:(label_dims, y_hat[...,i]) for i,key in enumerate(label_keys)}

    #make a dataset of the true and predicted patch data
    ds_return = xr.Dataset(data_vars = new_ds_dict)
    # ds_return = xr.Dataset(data_vars=dict(true_no_hail = (["n_samples", "lat_dim", "lon_dim"], output_array_np[:,:,:,0]),
    #                             true_hail = (["n_samples", "lat_dim", "lon_dim"], output_array_np[:,:,:,1]),
    #                             # true_sev_hail = (["patch", "x", "y"], output_array[:,:,:,2]),
    #                             # true_sig_sev_hail = (["patch", "x", "y"], output_array[:,:,:,3]),
    #                             predicted_no_hail = (["n_samples", "lat_dim", "lon_dim"], y_hat[:,:,:,0]),
    #                             predicted_hail = (["n_samples", "lat_dim", "lon_dim"], y_hat[:,:,:,1]),
    #                             comp_dz = (["n_samples", "lat_dim", "lon_dim"], comp_dz),
    #                             # predicted_sev_hail = (["patch", "x", "y"], y_hat[:,:,:,2]),
    #                             # predicted_sig_sev_hail = (["patch", "x", "y"], y_hat[:,:,:,3])
    #                             ),
    #                     coords=dict(lon_dim = range(output_array_np.shape[2]),
    #                             lat_dim = range(output_array_np.shape[1]),
    #                             lat = (["n_samples", "lat_dim", "lon_dim"], lats),
    #                             lon = (["n_samples", "lat_dim", "lon_dim"], lons),
    #                             n_samples = range(y_hat.shape[0])))

    #save out the prediction and truth values
    ds_return.to_netcdf(predictions_outfile)