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

    parser.add_argument('--single_checkpoint', type=str, default='/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/no_refl_trained_at_init_time_2022_08_03/saved_models/fold_0')
    parser.add_argument('--predictions_outfile', type=str, default='/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/no_refl_trained_at_init_time_2022_08_03/patches/predictions/y_hats.nc')
    parser.add_argument('--args_json', type=str, default='/home/tgschmidt/hail/configs/predict.json')
    parser.add_argument('--examples', type=str, default='/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/no_refl_trained_at_init_time_2022_08_03/patches/test/examples/*')
    parser.add_argument('--labels', type=str, default='/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/no_refl_trained_at_init_time_2022_08_03/patches/test/labels/*')
    
    return parser


if __name__ == "__main__":
    # Perform all the argument parsing
    parser = create_parser()
    args = parser.parse_args()
    with open(args.args_json) as json_file:
        json_dict = json.load(json_file)
    args = vars(args)
    args.update(json_dict)

    label_patches_dir = args["labels"]
    example_patches_dir = args["examples"]
    checkpoint_path = args["single_checkpoint"]
    predictions_outfile = args["predictions_outfile"]


    # glob files
    label_files = glob.glob(label_patches_dir)
    label_files.sort()
    example_files = glob.glob(example_patches_dir)
    example_files.sort()

    #open all the data in one xarray dataset
    examples_ds = xr.open_mfdataset(example_files, concat_dim='n_samples', combine='nested', engine='netcdf4', decode_cf=False)
    labels_ds = xr.open_mfdataset(label_files, concat_dim='n_samples', combine='nested', engine='netcdf4', decode_cf=False)

    label_keys = list(labels_ds.keys())
    label_dims = list(labels_ds.dims)

    if "hailcast" in examples_ds.keys():
        examples_ds = examples_ds.drop("hailcast")

    labels_ds.close()
    input_array = examples_ds.to_array()
    examples_ds.close()

    input_array = input_array.transpose("n_samples", ...)
    input_array = input_array.transpose(..., "variable")

    input_array_np = input_array.to_numpy()

    #read in the unet
    model = keras.models.load_model(checkpoint_path, compile=False)#, custom_objects={"max_csi": MaxCriticalSuccessIndex()})
        
    #evaluate the unet on the testing data
    y_hat = model.predict(input_array_np)

    new_ds_dict = {key:(label_dims, y_hat[...,i]) for i,key in enumerate(label_keys)}

    #make a dataset of the true and predicted patch data
    ds_return = xr.Dataset(data_vars = new_ds_dict)

    #save out the prediction and truth values
    ds_return.to_netcdf(predictions_outfile)