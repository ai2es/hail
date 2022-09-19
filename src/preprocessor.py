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

    parser.add_argument('--examples', type=str, default='/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/trained_at_init_time_2022_08_09/patches/train_val/examples/*')
    parser.add_argument('--labels', type=str, default='/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/trained_at_init_time_2022_08_09/patches/train_val/labels/*')
    parser.add_argument('--tf_ds_dir', type=str, default='/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/trained_at_init_time_2022_08_09/patches/train_val/tf_datasets')
    parser.add_argument('--feature_vars_to_drop', type=str, nargs='+', default=['lon', 'lat'])
    parser.add_argument('--label_vars_to_drop', type=str, nargs='+', default=['time', 'lon', 'lat', 'MESH95', 'MESH_class_bin_0'])
    parser.add_argument('--approx_file_clumping_num', type=int, default=8)
    parser.add_argument('--n_parallel_runs', type=int, default=15)
    parser.add_argument('--run_num', type=int, default=0)

    return parser


if __name__ == "__main__":
    # Perform all the argument parsing
    parser = create_parser()
    args = parser.parse_args()
    args = vars(args)

    netcdf_examples_dir = args["examples"]
    netcdf_labels_dir = args["labels"]
    tf_ds_dir = args["tf_ds_dir"]
    input_vars_to_drop = args["feature_vars_to_drop"]
    output_vars_to_drop = args["label_vars_to_drop"]
    approx_file_clumping_num = args["approx_file_clumping_num"]
    n_parallel_runs = args["n_parallel_runs"]
    run_num = args["run_num"]

    input_files = glob.glob(netcdf_examples_dir)
    input_files.sort()
    output_files = glob.glob(netcdf_labels_dir)
    output_files.sort()

    input_files = np.array_split(np.array(input_files), len(input_files)//approx_file_clumping_num)
    output_files = np.array_split(np.array(output_files), len(output_files)//approx_file_clumping_num)

    # Split into chunks for each parallel run
    input_files = np.array_split(np.array(input_files, dtype=object), n_parallel_runs)[run_num]
    output_files = np.array_split(np.array(output_files, dtype=object), n_parallel_runs)[run_num]

    for i, (input_file_clump, output_file_clump) in enumerate(zip(input_files, output_files)):
        start_time = time.time()
        input_ds_list = []
        output_ds_list = []
        for input_file, output_file in zip(input_file_clump, output_file_clump):
            input_ds_list.append(xr.open_dataset(input_file))
            output_ds_list.append(xr.open_dataset(output_file))

        # input_ds = xr.open_mfdataset(input_file_clump, concat_dim='n_samples', combine='nested', parallel=False, engine='netcdf4')
        # output_ds = xr.open_mfdataset(output_file_clump, concat_dim='n_samples', combine='nested', parallel=False, engine='netcdf4')

        input_ds = xr.concat(input_ds_list, dim="n_samples")
        output_ds = xr.concat(output_ds_list, dim="n_samples")

        for input_ds_old, output_ds_old in zip(input_ds_list, output_ds_list):
            input_ds_old.close()
            output_ds_old.close()

        output_ds = output_ds.drop(output_vars_to_drop)
        input_ds = input_ds.drop(input_vars_to_drop)

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

        #### TEMP ##########
        def min_max_norm(variable):
            """
            This does min_max scaling on the
            given variable.
            """
            if (np.max(variable) - np.min(variable)) == 0:
                return np.zeros(variable.shape)
            
            variable = (variable - np.min(variable)) / (np.max(variable) - np.min(variable))
            return variable
        
        output_array = output_array.to_numpy()
        input_array = input_array.to_numpy()
        print(np.max(input_array))
        for j in range(input_array.shape[3]):
            input_array[:,:,:,j] = min_max_norm(input_array[:,:,:,j])
        # input_array = np.apply_along_axis(min_max_norm, 3, input_array)
        print(np.max(input_array))
        print(input_array.shape)
        #######################

        tf_ds = tf.data.Dataset.from_tensor_slices((input_array, output_array))

        # input_array.close()
        # output_array.close()

        # NOTE: The save call below causes massive memory leaks. That is why we have to run this script in short bursts.
        # see https://github.com/tensorflow/tensorflow/issues/56177
        # and https://stackoverflow.com/questions/72308983/memory-leak-with-tensorflow-experimental-save
        tf_ds_name = str(run_num) + "-" + str(i) + ".tf"
        tf.data.experimental.save(tf_ds, os.path.join(tf_ds_dir, tf_ds_name))

        end_time = time.time()
        print("Done tf dataset file number " + str(run_num) + "-" + str(i) + ". Took " + str(end_time-start_time) + " seconds to complete.")