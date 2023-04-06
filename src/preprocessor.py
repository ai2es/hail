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

    parser.add_argument('--examples', type=str, default='/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour-128_size-more_fields-1_inch/patches/train/examples/*')
    parser.add_argument('--labels', type=str, default='/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour-128_size-more_fields-1_inch/patches/train/labels*')
    parser.add_argument('--output_ds_dir', type=str, default='/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour-128_size-more_fields-1_inch/patches/train/tf_datasets')
    parser.add_argument('--min_max_dir', type=str, default='/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour-more_fields-1_inch-cross_val/patches/cv_folds/fold_0000/mins_maxs')
    parser.add_argument('--feature_vars_to_drop', type=str, nargs='+', default=['lon', 'lat', 'time']) # Use to have hailcast
    parser.add_argument('--label_vars_to_drop', type=str, nargs='+', default=['time', 'lon', 'lat', 'MESH95'])
    parser.add_argument('--approx_file_clumping_num', type=int, default=None) # Was 3
    parser.add_argument('--n_parallel_runs', type=int, default=None) # Was 15
    parser.add_argument('--run_num', type=int, default=0)
    # parser.add_argument('--ne_dim_num', type=int, default=1)
    parser.add_argument('--selected_time', type=int, default=None)
    parser.add_argument('--ne_dim_size', type=int, default=18)
    parser.add_argument('--hailcast_threshold', type=float, default=1)
    parser.add_argument('--save_as_netcdf', '-s', action='store_true')
    parser.add_argument('--has_ne_dim', '-d', action='store_true')
    parser.add_argument('--min_max_normalize', '-m', action='store_true')
    parser.add_argument('--normalize_hailcast', '-n', action='store_true')
    parser.add_argument('--make_hailcast_probabilistic', '-p', action='store_true')
    parser.add_argument('--remove_patches_with_nans', '-r', action='store_true')
    parser.add_argument('--data_is_3D', '-t', action='store_true')
    parser.add_argument('--unpack_ne', '-u', action='store_true')
    # parser.add_argument('--unpack_3D', '-u', action='store_true')

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


def make_hailcast_probabilistic(hailcast, threshold, ne_dim_num, ne_dim_size):
    hailcast_class = np.zeros(hailcast.shape, dtype=np.float64)
    hailcast_class[np.nonzero(hailcast > threshold)] = 1

    hailcast_prob = np.sum(hailcast_class, axis=ne_dim_num)/ne_dim_size
    hailcast_prob = np.expand_dims(hailcast_prob, axis=ne_dim_num)
    hailcast_prob = np.repeat(hailcast_prob, ne_dim_size, axis=ne_dim_num) # TODO: Investigate if this breaks order

    return hailcast_prob


def unpack_ne_dim_input(variable, ne_dim_num):
    variable_shape_adjusted = list(variable.shape)
    ne_dim_size = variable_shape_adjusted.pop(ne_dim_num)
    variable_shape_adjusted[0] = variable_shape_adjusted[0] * ne_dim_size

    return np.reshape(variable, tuple(variable_shape_adjusted))


def unpack_ne_dim_output(variable, ne_dim_size, ne_dim_num):
    variable_shape_adjusted = list(variable.shape)
    variable = np.expand_dims(variable, axis=ne_dim_num)
    variable_repeated = np.repeat(variable, ne_dim_size, axis=ne_dim_num) # TODO: Investigate if this breaks order
    variable_shape_adjusted[0] = variable_shape_adjusted[0] * ne_dim_size

    return np.reshape(variable_repeated, tuple(variable_shape_adjusted)) # Only ok because we are making sure that the "ne" dim is next to the "n_samples" dim


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
    # ne_dim_num = args["ne_dim_num"]
    ne_dim_size = args["ne_dim_size"]
    hailcast_threshold = args["hailcast_threshold"]
    has_ne_dim = args["has_ne_dim"]
    min_max_normalize = args["min_max_normalize"]
    normalize_hailcast = args["normalize_hailcast"]
    make_hailcast_probabilistic_bool = args["make_hailcast_probabilistic"]
    remove_patches_with_nans = args["remove_patches_with_nans"]
    data_is_3D = args["data_is_3D"]
    selected_time = args["selected_time"]
    unpack_ne = args["unpack_ne"]

    input_files = glob.glob(netcdf_examples_dir)
    input_files.sort()
    output_files = glob.glob(netcdf_labels_dir)
    output_files.sort()

    sample_count = 0

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
            input_ds = xr.open_mfdataset(input_file_clump, concat_dim='n_samples', combine='nested', engine='netcdf4')
            output_ds = xr.open_mfdataset(output_file_clump, concat_dim='n_samples', combine='nested', engine='netcdf4')
        else:
            input_ds = xr.open_dataset(input_file_clump)
            output_ds = xr.open_dataset(output_file_clump)

        example_maxs = xr.open_dataset(os.path.join(min_max_dir, "examples_max.nc"))
        example_mins = xr.open_dataset(os.path.join(min_max_dir, "examples_min.nc"))
        label_maxs = xr.open_dataset(os.path.join(min_max_dir, "labels_max.nc"))
        label_mins = xr.open_dataset(os.path.join(min_max_dir, "labels_min.nc"))

        output_ds = output_ds.drop(output_vars_to_drop, errors="ignore")
        label_maxs = label_maxs.drop(output_vars_to_drop, errors="ignore")
        label_mins = label_mins.drop(output_vars_to_drop, errors="ignore")
        input_ds = input_ds.drop(input_vars_to_drop, errors="ignore")
        example_maxs = example_maxs.drop(input_vars_to_drop, errors="ignore")
        example_mins = example_mins.drop(input_vars_to_drop, errors="ignore")

        if data_is_3D:
            if has_ne_dim:
                input_ds = input_ds.transpose("n_samples", "ne", "lat_dim", "lon_dim", "time_dim")
                ne_dim_num = 1
            else:
                input_ds = input_ds.transpose("n_samples", "lat_dim", "lon_dim", "time_dim")
            output_ds = output_ds.transpose("n_samples", "lat_dim", "lon_dim", "time_dim")
        else:
            if has_ne_dim:
                input_ds = input_ds.transpose("n_samples", "ne", "lat_dim", "lon_dim")
                ne_dim_num = 1
            else:
                input_ds = input_ds.transpose("n_samples", "lat_dim", "lon_dim")
            output_ds = output_ds.transpose("n_samples", "lat_dim", "lon_dim")

        input_keys = list(input_ds.keys())
        output_keys = list(output_ds.keys())

        if data_is_3D and selected_time is not None:
            input_ds = input_ds[{"time_dim": selected_time}]
            output_ds = output_ds[{"time_dim": selected_time}]
        
        if "hailcast" in input_keys and has_ne_dim and make_hailcast_probabilistic_bool:
            if not unpack_ne:
                raise Exception("Cannot have no ne unapacking and make_hailcast_probabilistic_bool right now since the make_hailcast_probabilistic_bool function does np.repeat stuff.")
            input_ds["hailcast"] = (input_ds["hailcast"].dims, make_hailcast_probabilistic(input_ds["hailcast"].data, hailcast_threshold, ne_dim_num, ne_dim_size))

        if has_ne_dim and unpack_ne:
            # Because we have to reshape dimensions
            new_input_ds = xr.Dataset()
            new_output_ds = xr.Dataset()

            for key in output_keys:
                new_output_ds = new_output_ds.assign({key: (output_ds[key].dims, unpack_ne_dim_output(output_ds[key].data, ne_dim_size, ne_dim_num))})
            output_ds.close()
            output_ds = new_output_ds

            for key in input_keys:
                new_dim = list(input_ds[key].dims)
                new_dim.pop(ne_dim_num)
                new_dim = tuple(new_dim)
                new_input_ds = new_input_ds.assign({key: (new_dim, unpack_ne_dim_input(input_ds[key].data, ne_dim_num))})
            input_ds.close()
            input_ds = new_input_ds

        if min_max_normalize:
            for key in input_keys:
                if key == "hailcast" and not normalize_hailcast:
                    continue
                input_ds[key] = (input_ds[key].dims, min_max_norm(input_ds[key].data, example_mins[key].data, example_maxs[key].data))

        if remove_patches_with_nans:
            count_before_nan_removal = input_ds[input_keys[0]].to_numpy().shape[0]
            merged_ds = xr.merge([input_ds, output_ds])
            merged_array = merged_ds.to_array()
            merged_ds.close()
            merged_ds = None
            merged_array = merged_array.transpose("n_samples", ...)
            merged_np = merged_array.to_numpy()
            merged_array.close()
            merged_array = None

            indeces_of_nans = np.isnan(merged_np)
            patches_with_no_nans = np.nonzero(np.any(indeces_of_nans, axis=tuple(np.arange(1,len(merged_np.shape)))) == 0)

            merged_np = None

            input_ds = input_ds[{"n_samples": patches_with_no_nans[0]}]
            output_ds = output_ds[{"n_samples": patches_with_no_nans[0]}]
            print("Nan section removed: " + str(count_before_nan_removal - input_ds[input_keys[0]].to_numpy().shape[0]))
        
        sample_count = sample_count + input_ds[input_keys[0]].to_numpy().shape[0]
                
        if save_as_netcdf:
            input_ds_name = "examples/" + file_name_num + ".nc"
            output_ds_name = "labels/" + file_name_num + ".nc"
            input_ds.to_netcdf(os.path.join(output_ds_dir, input_ds_name))
            output_ds.to_netcdf(os.path.join(output_ds_dir, output_ds_name))
            output_ds.close()
            input_ds.close()

        else:
            if "hailcast" in input_keys and not normalize_hailcast:
                input_ds = input_ds.drop("hailcast")
            output_array = output_ds.to_array()
            input_array = input_ds.to_array()
            output_ds.close()
            input_ds.close()

            output_array = output_array.astype("float32") # TODO: Make this a setting later?
            input_array = input_array.astype("float32")

            input_array = input_array.transpose(..., "variable")
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
        print("It contained this many patches: " + str(sample_count))
