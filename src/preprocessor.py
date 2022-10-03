import xarray as xr
import tensorflow as tf
import os
import argparse
import glob
import numpy as np
import time


def create_parser():
    '''
    Create argument parser
    '''
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description='Unet Preprocessing', fromfile_prefix_chars='@')

    parser.add_argument('--examples', type=str, default='/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/trained_at_init_time/patches/test/init_plus_60/unprocessed/examples/*')
    parser.add_argument('--labels', type=str, default='/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/trained_at_init_time/patches/test/init_plus_60/unprocessed/labels/*')
    parser.add_argument('--output_ds_dir', type=str, default='/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/trained_at_init_time/patches/test/init_plus_60/processed')
    parser.add_argument('--min_max_dir', type=str, default='/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/trained_at_init_time/patches/mins_maxs')
    parser.add_argument('--feature_vars_to_drop', type=str, nargs='+', default=['lon', 'lat', 'hailcast'])
    parser.add_argument('--label_vars_to_drop', type=str, nargs='+', default=['time', 'lon', 'lat', 'MESH95'])
    parser.add_argument('--approx_file_clumping_num', type=int, default=None) # Was 8
    parser.add_argument('--n_parallel_runs', type=int, default=None) # Was 15
    parser.add_argument('--run_num', type=int, default=0)
    parser.add_argument('--save_as_netcdf', '-n', action='store_true')

    return parser


def min_max_norm(variable, var_min, var_max):
    """
    This does min_max scaling on the
    given variable.
    """
    if (var_max - var_min) == 0:
        return np.zeros(variable.shape)
    
    variable = (variable - var_min) / (var_max - var_min)
    return variable


if __name__ == "__main__":
    # Perform all the argument parsing
    parser = create_parser()
    args = parser.parse_args()
    args = vars(args)

    netcdf_examples_dir = args["examples"]
    netcdf_labels_dir = args["labels"]
    output_ds_dir = args["output_ds_dir"]
    min_max_dir = args["min_max_dir"]
    input_vars_to_drop = args["feature_vars_to_drop"]
    output_vars_to_drop = args["label_vars_to_drop"]
    approx_file_clumping_num = args["approx_file_clumping_num"]
    n_parallel_runs = args["n_parallel_runs"]
    run_num = args["run_num"]
    save_as_netcdf = args["save_as_netcdf"]

    input_files = glob.glob(netcdf_examples_dir)
    input_files.sort()
    output_files = glob.glob(netcdf_labels_dir)
    output_files.sort()

    if approx_file_clumping_num is not None:
        input_files = np.array_split(np.array(input_files), len(input_files)//approx_file_clumping_num)
        output_files = np.array_split(np.array(output_files), len(output_files)//approx_file_clumping_num)

    if n_parallel_runs is not None:
        # Split into chunks for each parallel run
        input_files = np.array_split(np.array(input_files, dtype=object), n_parallel_runs)[run_num]
        output_files = np.array_split(np.array(output_files, dtype=object), n_parallel_runs)[run_num]

    for i, (input_file_clump, output_file_clump) in enumerate(zip(input_files, output_files)):
        file_name_num = "{:04d}".format(run_num) + "-" + "{:04d}".format(i)
        start_time = time.time()
        
        if approx_file_clumping_num is not None:
            input_ds = xr.open_mfdataset(input_file_clump, concat_dim='n_samples', combine='nested', engine='netcdf4', decode_cf=False)
            output_ds = xr.open_mfdataset(output_file_clump, concat_dim='n_samples', combine='nested', engine='netcdf4', decode_cf=False)
        else:
            input_ds = xr.open_dataset(input_file_clump, decode_cf=False)
            output_ds = xr.open_dataset(output_file_clump, decode_cf=False)

        example_maxs = xr.open_dataset(os.path.join(min_max_dir, "examples_max.nc"), decode_cf=False)
        example_mins = xr.open_dataset(os.path.join(min_max_dir, "examples_min.nc"), decode_cf=False)
        label_maxs = xr.open_dataset(os.path.join(min_max_dir, "labels_max.nc"), decode_cf=False)
        label_mins = xr.open_dataset(os.path.join(min_max_dir, "labels_min.nc"), decode_cf=False)

        output_ds = output_ds.drop(output_vars_to_drop)
        label_maxs = label_maxs.drop(output_vars_to_drop)
        label_mins = label_mins.drop(output_vars_to_drop)
        input_ds = input_ds.drop(input_vars_to_drop)
        example_maxs = example_maxs.drop(input_vars_to_drop)
        example_mins = example_mins.drop(input_vars_to_drop)

        input_keys = list(input_ds.keys())
        output_keys = list(output_ds.keys())

        for key in input_keys:
            input_ds[key] = (input_ds[key].dims, min_max_norm(input_ds[key], example_mins[key], example_maxs[key]).data)
        # for key in output_keys:
        #     output_ds[key] = (output_ds[key].dims, min_max_norm(output_ds[key], label_mins[key], label_maxs[key]).data)

        if save_as_netcdf:
            input_ds_name = "examples/" + file_name_num + ".nc"
            output_ds_name = "labels/" + file_name_num + ".nc"
            input_ds.to_netcdf(os.path.join(output_ds_dir, input_ds_name))
            output_ds.to_netcdf(os.path.join(output_ds_dir, output_ds_name))
            output_ds.close()
            input_ds.close()

        else:
            output_array = output_ds.to_array()
            input_array = input_ds.to_array()
            output_ds.close()
            input_ds.close()

            output_array = output_array.astype("float32") # TODO: Make this a setting later?
            input_array = input_array.astype("float32")

            input_array = input_array.transpose("n_samples", ...)
            input_array = input_array.transpose(..., "variable")

            output_array = output_array.transpose("n_samples", ...)
            output_array = output_array.transpose(..., "variable")

            tf_ds = tf.data.Dataset.from_tensor_slices((input_array, output_array))

            input_array.close()
            output_array.close()

            # NOTE: The save call below causes massive memory leaks. That is why we have to run this script in short bursts.
            # see https://github.com/tensorflow/tensorflow/issues/56177
            # and https://stackoverflow.com/questions/72308983/memory-leak-with-tensorflow-experimental-save
            tf_ds_name = file_name_num + ".tf"
            tf.data.experimental.save(tf_ds, os.path.join(output_ds_dir, tf_ds_name))

        end_time = time.time()
        print("Done dataset file number " + file_name_num + ". Took " + str(end_time-start_time) + " seconds to complete.")