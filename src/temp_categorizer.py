import xarray as xr
import glob
import os
import argparse
import numpy as np
import tensorflow as tf

def create_parser():
    '''
    Create argument parser
    '''
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description='Unet Preprocessing', fromfile_prefix_chars='@')

    parser.add_argument('--examples', type=str, default='/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/trained_at_init_time_4_class_2022_08_09/patches/train_val/examples/*')
    parser.add_argument('--labels', type=str, default='/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/trained_at_init_time_4_class_2022_08_09/patches/train_val/labels/*')
    parser.add_argument('--tf_ds_dir', type=str, default='/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/trained_at_init_time_4_class_2022_08_09/patches/train_val/tf_datasets')
    parser.add_argument('--feature_vars_to_drop', type=str, nargs='+', default=['lon', 'lat'])
    parser.add_argument('--label_vars_to_drop', type=str, nargs='+', default=['time', 'lon', 'lat', 'MESH95'])
    parser.add_argument('--approx_file_clumping_num', type=int, default=5)

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

    input_files = glob.glob(netcdf_examples_dir)
    input_files.sort()
    output_files = glob.glob(netcdf_labels_dir)
    output_files.sort()

    input_files = np.array_split(np.array(input_files), len(input_files)//approx_file_clumping_num)
    output_files = np.array_split(np.array(output_files), len(output_files)//approx_file_clumping_num)

    for i, (input_file_clump, output_file_clump) in enumerate(zip(input_files, output_files)):
        print(i)
        input_ds_list = []
        output_ds_list = []
        for input_file, output_file in zip(input_file_clump, output_file_clump):
            input_ds_list.append(xr.open_dataset(input_file))
            output_ds_list.append(xr.open_dataset(output_file))

        input_ds = xr.concat(input_ds_list, dim="n_samples")
        output_ds = xr.concat(output_ds_list, dim="n_samples")
        output_ds = output_ds.drop(output_vars_to_drop)
        input_ds = input_ds.drop(input_vars_to_drop)

        ####################################################################
        output_array = output_ds.to_array()
        input_array = input_ds.to_array()

        output_array = output_array.astype("float32") # TODO: Make this a setting later?
        input_array = input_array.astype("float32")

        input_array = input_array.transpose("n_samples", ...)
        input_array = input_array.transpose(..., "variable")

        output_array = output_array.transpose("n_samples", ...)
        output_array = output_array.transpose(..., "variable")

        tf_ds = tf.data.Dataset.from_tensor_slices((input_array, output_array))

        tf_ds_name = str(i) + ".tf"
        tf.data.experimental.save(tf_ds, os.path.join("/scratch/tgschmidt", tf_ds_name))

        print("derp")

    # for i in range(30):
    #     for j, (input_file, output_file) in enumerate(zip(input_files, output_files)):
    #         derp = xr.open_dataset(input_file)
    #         derp2 = xr.open_dataset(output_file)

    #         print((i,j))

    