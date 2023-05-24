import xarray as xr
import numpy as np
import glob
import os
import argparse


EXAMPLE_VARS_TO_DROP = ["time", "time_0", "time_1", "time_2", "time_3", "time_4", "time_5", "time_6", "time_7", "time_8", "time_9", "time_10", "time_11", "time_12"]
LABEL_VARS_TO_DROP = ["time", "time_0", "time_1", "time_2", "time_3", "time_4", "time_5", "time_6", "time_7", "time_8", "time_9", "time_10", "time_11", "time_12"]


def create_parser():
    '''
    Create argument parser
    '''
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description='Data Splitting', fromfile_prefix_chars='@')

    parser.add_argument('--examples_glob', type=str, default="/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour-more_fields-1_inch-cross_val/patches/cv_folds/fold_0004/train/examples/*")
    parser.add_argument('--labels_glob', type=str, default="/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour-more_fields-1_inch-cross_val/patches/cv_folds/fold_0004/train/labels/*")
    parser.add_argument('--min_maxs_dir', type=str, default="/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour-more_fields-1_inch-cross_val/patches/cv_folds/fold_0004/mins_maxs")
    parser.add_argument('--2d_time_naming', '-n', action='store_true')

    return parser


def var_name_trimmer(var_name):
    while var_name[-1] != "_":
        var_name = var_name[:-1]
    
    return var_name[:-1]


# Perform all the argument parsing
parser = create_parser()
args = parser.parse_args()
args = vars(args)

examples_files = glob.glob(args["examples_glob"])
examples_files.sort()
labels_files = glob.glob(args["labels_glob"])
labels_files.sort()

one_examples_ds = xr.open_dataset(examples_files[0])
one_examples_ds = one_examples_ds.drop(EXAMPLE_VARS_TO_DROP, errors="ignore")
example_keys = [key for key in one_examples_ds.keys()]
one_examples_ds.close()

one_labels_ds = xr.open_dataset(labels_files[0])
one_labels_ds = one_labels_ds.drop(LABEL_VARS_TO_DROP, errors="ignore")
label_keys = [key for key in one_labels_ds.keys()]
one_labels_ds.close()

examples_max = np.full(len(example_keys), -np.inf)
labels_max = np.full(len(label_keys), -np.inf)
examples_min = np.full(len(example_keys), np.inf)
labels_min = np.full(len(label_keys), np.inf)

for j, (examples_file, labels_file) in enumerate(zip(examples_files, labels_files)):
    examples_ds = xr.open_dataset(examples_file)
    labels_ds = xr.open_dataset(labels_file)

    examples_ds = examples_ds.drop(EXAMPLE_VARS_TO_DROP, errors="ignore")
    labels_ds = labels_ds.drop(LABEL_VARS_TO_DROP, errors="ignore")

    examples_array = examples_ds.to_array().to_numpy()
    labels_array = labels_ds.to_array().to_numpy()

    examples_ds.close()
    labels_ds.close()

    for i in range(len(examples_max)):
        examples_max[i] = np.nanmax((examples_max[i], np.nanmax(examples_array[i,...])))
        examples_min[i] = np.nanmin((examples_min[i], np.nanmin(examples_array[i,...])))

    for i in range(len(labels_max)):
        labels_max[i] = np.nanmax((labels_max[i], np.nanmax(labels_array[i,...])))
        labels_min[i] = np.nanmin((labels_min[i], np.nanmin(labels_array[i,...])))

    print("Finished file num: " + str(j))

if args["2d_time_naming"]:
    base_examples_names = np.array([var_name_trimmer(names) for names in example_keys])
    base_labels_names = np.array([var_name_trimmer(names) for names in label_keys])
    base_examples_unique_names = np.unique(base_examples_names)
    base_labels_unique_names = np.unique(base_labels_names)

    for unique_example_name in base_examples_unique_names:
        examples_max[np.nonzero(base_examples_names == unique_example_name)] = np.nanmax(examples_max[np.nonzero(base_examples_names == unique_example_name)])
        examples_min[np.nonzero(base_examples_names == unique_example_name)] = np.nanmin(examples_min[np.nonzero(base_examples_names == unique_example_name)])
        
    for unique_label_name in base_labels_unique_names:
        labels_max[np.nonzero(base_labels_names == unique_label_name)] = np.nanmax(labels_max[np.nonzero(base_labels_names == unique_label_name)])
        labels_min[np.nonzero(base_labels_names == unique_label_name)] = np.nanmin(labels_min[np.nonzero(base_labels_names == unique_label_name)])

examples_max_ds = xr.Dataset(data_vars=dict(zip(example_keys, examples_max)))
examples_min_ds = xr.Dataset(data_vars=dict(zip(example_keys, examples_min)))
labels_max_ds = xr.Dataset(data_vars=dict(zip(label_keys, labels_max)))
labels_min_ds = xr.Dataset(data_vars=dict(zip(label_keys, labels_min)))

examples_max_ds.to_netcdf(os.path.join(args["min_maxs_dir"], "examples_max.nc"))
examples_min_ds.to_netcdf(os.path.join(args["min_maxs_dir"], "examples_min.nc"))
labels_max_ds.to_netcdf(os.path.join(args["min_maxs_dir"], "labels_max.nc"))
labels_min_ds.to_netcdf(os.path.join(args["min_maxs_dir"], "labels_min.nc"))