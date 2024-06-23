import xarray as xr
import glob
from tensorflow import keras
import argparse
import os
import numpy as np


def create_parser():
    '''
    Create argument parser
    '''
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description='Unet Model', fromfile_prefix_chars='@')

    parser.add_argument('--single_checkpoint', type=str, default='/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour-128_size-more_fields-1_inch/saved_models/checkpoints/123')
    parser.add_argument('--predictions_outfile', type=str, default='/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour-128_size-more_fields-1_inch/patches/animations/20190501/predictions/y_hats.nc')
    parser.add_argument('--examples', type=str, default='/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour-128_size-more_fields-1_inch/patches/animations/20190501/processed/examples/*')
    parser.add_argument('--labels', type=str, default='/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour-128_size-more_fields-1_inch/patches/animations/20190501/processed/labels/*')
    parser.add_argument('--selected_time', type=int, default=None)
    parser.add_argument('--n_parallel_runs', type=int, default=None)
    parser.add_argument('--ens_member', type=int, default=None)
    parser.add_argument('--use_hailcast', '-u', action='store_true')
    parser.add_argument('--multi_model', '-m', action='store_true')
    parser.add_argument('--run_num', type=int, default=0)
    
    return parser


if __name__ == "__main__":
    # Perform all the argument parsing
    parser = create_parser()
    args = parser.parse_args()
    args = vars(args)

    label_patches_dir = args["labels"]
    example_patches_dir = args["examples"]
    checkpoint_path = args["single_checkpoint"]
    predictions_outfile = args["predictions_outfile"]
    selected_time = args["selected_time"]
    use_hailcast = args["use_hailcast"]
    run_num = args["run_num"]
    multi_model = args["multi_model"]
    n_parallel_runs = args["n_parallel_runs"]
    ens_member = args["ens_member"]

    TIMES_IN_MINS = ["00","05","10","15","20","25","30","35","40","45","50","55"]
    CHOSEN_MODELS = ["23"]
    MODEL_INDICES = [0,0,0,0,0,0,0,0,0,0,0,0]

    if multi_model:
        selected_time = run_num
        checkpoint_path = os.path.join(checkpoint_path, CHOSEN_MODELS[MODEL_INDICES[run_num]])

    # glob files
    label_files = glob.glob(label_patches_dir)
    label_files.sort()
    example_files = glob.glob(example_patches_dir)
    example_files.sort()

    # Do my usual splitting for multiple jobs
    if n_parallel_runs is not None:
        example_files = np.array_split(np.array(example_files), n_parallel_runs)[run_num]
        label_files = np.array_split(np.array(label_files), n_parallel_runs)[run_num]

    #open all the data in one xarray dataset
    examples_ds = xr.open_mfdataset(example_files, concat_dim='n_samples', combine='nested', engine='netcdf4')
    labels_ds = xr.open_mfdataset(label_files, concat_dim='n_samples', combine='nested', engine='netcdf4')

    if "time_dim" in examples_ds.dims:
        examples_ds = examples_ds.transpose("n_samples", "lat_dim", "lon_dim", "time_dim", ...)
        labels_ds = labels_ds.transpose("n_samples", "lat_dim", "lon_dim", "time_dim", ...)
        if selected_time is not None:
            examples_ds = examples_ds[{"time_dim": selected_time}]
            labels_ds = labels_ds[{"time_dim": selected_time}]
    else:
        examples_ds = examples_ds.transpose("n_samples", "lat_dim", "lon_dim", ...)
        labels_ds = labels_ds.transpose("n_samples", "lat_dim", "lon_dim", ...)
    
    #read in the unet
    model = keras.models.load_model(checkpoint_path, compile=False)

    if "ne" in examples_ds.dims:
        if ens_member is not None:
            examples_ds = examples_ds[{"ne": ens_member}]

            if "hailcast" in examples_ds.keys() and not use_hailcast:
                examples_ds = examples_ds.drop("hailcast")
            
            input_array = examples_ds.to_array()
            examples_ds.close()
            examples_ds = None
            input_array = input_array.transpose(..., "variable").to_numpy()

            #evaluate the unet on the testing data
            y_hat = model.predict(input_array)
            input_array = None
        
        else:
            full_ensemble = []
            for i in range(examples_ds.dims["ne"]):
                member_example_ds = examples_ds[{"ne": i}]

                if "hailcast" in member_example_ds.keys() and not use_hailcast:
                    member_example_ds = member_example_ds.drop("hailcast")

                input_array = member_example_ds.to_array()
                member_example_ds.close()
                member_example_ds = None
                input_array = input_array.transpose(..., "variable").to_numpy()

                #evaluate the unet on the testing data
                full_ensemble.append(model.predict(input_array))

            full_ensemble = np.stack(full_ensemble)
            y_hat = np.mean(full_ensemble, axis=0)
            full_ensemble = None
    else:
        if "hailcast" in examples_ds.keys() and not use_hailcast:
            examples_ds = examples_ds.drop("hailcast")

        input_array = examples_ds.to_array()
        examples_ds.close()
        examples_ds = None
        input_array = input_array.transpose(..., "variable").to_numpy()

        #evaluate the unet on the testing data
        y_hat = model.predict(input_array)
        input_array = None

    label_keys = list(labels_ds.keys())
    label_dims = list(labels_ds.dims)
    labels_ds.close()
    new_ds_dict = {key:(label_dims, y_hat[...,i]) for i,key in enumerate(label_keys)}
    y_hat = None

    #make a dataset of the true and predicted patch data
    ds_return = xr.Dataset(data_vars = new_ds_dict)
    new_ds_dict = None
    print(ds_return)

    #save out the prediction and truth values
    if multi_model:
        if ".nc" in predictions_outfile:
            raise Exception("Given outfile path is not valid. For multi-model case it must be a directory instead of file name with .nc")
        predictions_outfile = os.path.join(predictions_outfile, "y_hats_" + TIMES_IN_MINS[run_num] + ".nc")

    elif n_parallel_runs is not None:
        if ".nc" in predictions_outfile:
            raise Exception("Given outfile path is not valid. For parallel runs case it must be a directory instead of file name with .nc")
        file_name_num = "{:04d}".format(run_num)
        predictions_outfile = os.path.join(predictions_outfile, "y_hats_" + file_name_num + ".nc")

    ds_return.to_netcdf(predictions_outfile)