import numpy as np
import os 
import glob
import xarray as xr
from sklearn.cluster import DBSCAN
from sklearn.model_selection import StratifiedGroupKFold
import argparse


SECOND_DS_SIZE = 0.2
EXAMPLES_GLOB = "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-1_hour-1_inch-train_val_test-cross_val/patches/trainvaltest/examples/*"
LABELS_GLOB = "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-1_hour-1_inch-train_val_test-cross_val/patches/trainvaltest/labels/*"
FIRST_DS_OUTPUT = "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-1_hour-1_inch-train_val_test-cross_val/patches/trainval"
SECOND_DS_OUTPUT = "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-1_hour-1_inch-train_val_test-cross_val/patches/test/unprocessed"
NUM_OUTPUT_FILES = (60,20)
N_FOLDS = 5
FOLD_PATH = "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour-more_fields-1_inch-cross_val/patches/cv_folds"


def cluster_by_storm_event(labels):
    labels = labels[{"time_dim": 0}] # Select only the first time (init time)

    sample_init_times = labels["time"].to_numpy()
    sample_init_times = sample_init_times.reshape(-1,1) # To fit (n_samples,1) convention needed for DBSCAN

    clustering = DBSCAN(eps=2.16e13, min_samples=1).fit(sample_init_times)

    return clustering.labels_


# Only for functions where I am stratifying
def get_group_positions(labels, storm_groups, second_ds_size_or_n_folds, is_cross_val=False):
    if is_cross_val:
        if type(second_ds_size_or_n_folds) is not int:
            raise Exception("For cross validation case you must give an int to represent n_folds.")
    else:
        if type(second_ds_size_or_n_folds) is not float:
            raise Exception("For the single dataset splitting case you must give a float to represent the size of the second dataset.")

    labels = labels["MESH_class_bin"].drop(["time","lon","lat"]).to_numpy()

    for i in np.arange(1,len(labels.shape)):
        storm_groups = np.expand_dims(storm_groups, -1)
    storm_groups = np.tile(storm_groups, labels.shape[1:])

    labels = labels.ravel()
    storm_groups = storm_groups.ravel()
    fake_X = np.zeros(len(labels))

    if is_cross_val:
        sgkf = StratifiedGroupKFold(n_splits=second_ds_size_or_n_folds)
    else:
        sgkf = StratifiedGroupKFold(n_splits=int(1/second_ds_size_or_n_folds))

    all_storm_groups = []
    for first_ds_indices, second_ds_indices in sgkf.split(fake_X, labels, storm_groups):
        all_storm_groups.append((np.unique(storm_groups[first_ds_indices]), np.unique(storm_groups[second_ds_indices])))

    if is_cross_val:
        return all_storm_groups
    else:
        # 0 index for one "fold"
        first_ds_indices = all_storm_groups[0][0]
        second_ds_indices = all_storm_groups[0][1]

        return np.unique(storm_groups[first_ds_indices]), np.unique(storm_groups[second_ds_indices])


def stratify_datasets_by_storm_events_once(examples_glob, labels_glob, first_ds_output_path, second_ds_output_path, num_output_files, second_ds_size):
    file_list = glob.glob(examples_glob)
    file_list.sort()
    examples_ds = xr.open_mfdataset(file_list, concat_dim='n_samples', combine='nested', engine='netcdf4')

    file_list = glob.glob(labels_glob)
    file_list.sort()
    label_ds = xr.open_mfdataset(file_list, concat_dim='n_samples', combine='nested', engine='netcdf4')

    storm_event_clusters = cluster_by_storm_event(label_ds)
    groups_in_first_ds, groups_in_second_ds = get_group_positions(label_ds,storm_event_clusters,second_ds_size,is_cross_val=False)

    first_ds_mask = np.zeros(len(storm_event_clusters), dtype=np.int64)
    second_ds_mask = np.zeros(len(storm_event_clusters), dtype=np.int64)

    for group in groups_in_first_ds:
        first_ds_mask[np.nonzero(storm_event_clusters == group)] = 1
    for group in groups_in_second_ds:
        second_ds_mask[np.nonzero(storm_event_clusters == group)] = 1

    first_example_ds = examples_ds[{"n_samples": np.nonzero(first_ds_mask)[0]}]
    first_labels_ds = label_ds[{"n_samples": np.nonzero(first_ds_mask)[0]}]
    second_example_ds = examples_ds[{"n_samples": np.nonzero(second_ds_mask)[0]}]
    second_labels_ds = label_ds[{"n_samples": np.nonzero(second_ds_mask)[0]}]

    split_verification(first_labels_ds, second_labels_ds, groups_in_first_ds, groups_in_second_ds)

    divded_indices_for_first_dataset_netcdfs = np.array_split(np.arange(first_example_ds.dims["n_samples"]), num_output_files)
    divded_indices_for_second_dataset_netcdfs = np.array_split(np.arange(second_example_ds.dims["n_samples"]), num_output_files)

    for i, (first_indices, second_indices) in enumerate(zip(divded_indices_for_first_dataset_netcdfs, divded_indices_for_second_dataset_netcdfs)):
        file_name_num = "{:04d}".format(i)

        first_example_ds[{"n_samples": first_indices}].to_netcdf(os.path.join(first_ds_output_path, "examples/" + file_name_num + ".nc"))
        first_labels_ds[{"n_samples": first_indices}].to_netcdf(os.path.join(first_ds_output_path, "labels/" + file_name_num + ".nc"))
        second_example_ds[{"n_samples": second_indices}].to_netcdf(os.path.join(second_ds_output_path, "examples/" + file_name_num + ".nc"))
        second_labels_ds[{"n_samples": second_indices}].to_netcdf(os.path.join(second_ds_output_path, "labels/" + file_name_num + ".nc"))


def stratified_kfold_cross_val(examples_glob, labels_glob, fold_path, num_output_files, n_folds):
    file_list = glob.glob(examples_glob)
    file_list.sort()
    examples_ds = xr.open_mfdataset(file_list, concat_dim='n_samples', combine='nested', engine='netcdf4')

    file_list = glob.glob(labels_glob)
    file_list.sort()
    label_ds = xr.open_mfdataset(file_list, concat_dim='n_samples', combine='nested', engine='netcdf4')

    storm_event_clusters = cluster_by_storm_event(label_ds)
    all_storm_groups = get_group_positions(label_ds,storm_event_clusters,n_folds,is_cross_val=True)

    for fold_num in range(n_folds):
        groups_in_train_ds = all_storm_groups[fold_num][0]
        groups_in_val_ds = all_storm_groups[fold_num][1]

        train_mask = np.zeros(len(storm_event_clusters), dtype=np.int64)
        val_mask = np.zeros(len(storm_event_clusters), dtype=np.int64)

        for group in groups_in_train_ds:
            train_mask[np.nonzero(storm_event_clusters == group)] = 1
        for group in groups_in_val_ds:
            val_mask[np.nonzero(storm_event_clusters == group)] = 1

        train_example_ds = examples_ds[{"n_samples": np.nonzero(train_mask)[0]}]
        train_labels_ds = label_ds[{"n_samples": np.nonzero(train_mask)[0]}]
        val_example_ds = examples_ds[{"n_samples": np.nonzero(val_mask)[0]}]
        val_labels_ds = label_ds[{"n_samples": np.nonzero(val_mask)[0]}]

        split_verification(train_labels_ds, val_labels_ds, groups_in_train_ds, groups_in_val_ds)

        divded_indices_for_train_netcdfs = np.array_split(np.arange(train_example_ds.dims["n_samples"]), num_output_files)
        divded_indices_for_val_netcdfs = np.array_split(np.arange(val_example_ds.dims["n_samples"]), num_output_files)

        fold_dir_name = "fold_" + "{:04d}".format(fold_num)
        fold_num_path = os.path.join(fold_path, fold_dir_name)
        train_ds_output_path = os.path.join(fold_num_path, "train")
        val_ds_output_path = os.path.join(fold_num_path, "val")

        for i, (train_indices, val_indices) in enumerate(zip(divded_indices_for_train_netcdfs, divded_indices_for_val_netcdfs)):
            file_name_num = "{:04d}".format(i)

            train_example_ds[{"n_samples": train_indices}].to_netcdf(os.path.join(train_ds_output_path, "examples/" + file_name_num + ".nc"))
            train_labels_ds[{"n_samples": train_indices}].to_netcdf(os.path.join(train_ds_output_path, "labels/" + file_name_num + ".nc"))
            val_example_ds[{"n_samples": val_indices}].to_netcdf(os.path.join(val_ds_output_path, "examples/" + file_name_num + ".nc"))
            val_labels_ds[{"n_samples": val_indices}].to_netcdf(os.path.join(val_ds_output_path, "labels/" + file_name_num + ".nc"))
        

def split_dataset_no_stratification(examples_glob, labels_glob, first_ds_output_path, second_ds_output_path, num_output_files, second_ds_size):
    if type(num_output_files) is not tuple or len(num_output_files) != 2:
        raise Exception("num_output_files must be tupled with 2 elements for the number of files in each output dataset.")
    
    file_list = glob.glob(examples_glob)
    file_list.sort()
    examples_ds = xr.open_mfdataset(file_list, concat_dim='n_samples', combine='nested', engine='netcdf4')

    file_list = glob.glob(labels_glob)
    file_list.sort()
    label_ds = xr.open_mfdataset(file_list, concat_dim='n_samples', combine='nested', engine='netcdf4')

    storm_event_clusters = cluster_by_storm_event(label_ds)
    num_of_storm_events = len(np.unique(storm_event_clusters))

    groups_in_second_ds = np.random.choice(num_of_storm_events, int(num_of_storm_events*second_ds_size), replace=False)
    groups_in_first_ds = np.setdiff1d(np.unique(storm_event_clusters), groups_in_second_ds)

    first_ds_mask = np.zeros(len(storm_event_clusters), dtype=np.int64)
    second_ds_mask = np.zeros(len(storm_event_clusters), dtype=np.int64)

    for group in groups_in_first_ds:
        first_ds_mask[np.nonzero(storm_event_clusters == group)] = 1
    for group in groups_in_second_ds:
        second_ds_mask[np.nonzero(storm_event_clusters == group)] = 1

    first_example_ds = examples_ds[{"n_samples": np.nonzero(first_ds_mask)[0]}]
    first_labels_ds = label_ds[{"n_samples": np.nonzero(first_ds_mask)[0]}]
    second_example_ds = examples_ds[{"n_samples": np.nonzero(second_ds_mask)[0]}]
    second_labels_ds = label_ds[{"n_samples": np.nonzero(second_ds_mask)[0]}]

    split_verification(first_labels_ds, second_labels_ds, groups_in_first_ds, groups_in_second_ds)

    divded_indices_for_first_dataset_netcdfs = np.array_split(np.arange(first_example_ds.dims["n_samples"]), num_output_files[0])
    divded_indices_for_second_dataset_netcdfs = np.array_split(np.arange(second_example_ds.dims["n_samples"]), num_output_files[1])

    for i, first_indices in enumerate(divded_indices_for_first_dataset_netcdfs):
        file_name_num = "{:04d}".format(i)

        first_example_ds[{"n_samples": first_indices}].to_netcdf(os.path.join(first_ds_output_path, "examples/" + file_name_num + ".nc"))
        first_labels_ds[{"n_samples": first_indices}].to_netcdf(os.path.join(first_ds_output_path, "labels/" + file_name_num + ".nc"))

    for i, second_indices in enumerate(divded_indices_for_second_dataset_netcdfs):
        file_name_num = "{:04d}".format(i)

        second_example_ds[{"n_samples": second_indices}].to_netcdf(os.path.join(second_ds_output_path, "examples/" + file_name_num + ".nc"))
        second_labels_ds[{"n_samples": second_indices}].to_netcdf(os.path.join(second_ds_output_path, "labels/" + file_name_num + ".nc"))


def split_verification(first_ds_labels, second_ds_labels, first_ds_groups, second_ds_groups):
    # Select only the first time (init time) to verify 
    first_ds_datetimes = first_ds_labels[{"time_dim": 0}]["time"].to_numpy()
    for second_ds_datetime in second_ds_labels[{"time_dim": 0}]["time"].to_numpy():
        if np.any(np.logical_and(first_ds_datetimes > second_ds_datetime - np.timedelta64(6,'h'), first_ds_datetimes <  second_ds_datetime + np.timedelta64(6,'h'))):
            raise Exception("Looks like the DBSCAN grouping failed. Investigate.")
    
    print("First dataset number of unique storm days: " + str(len(np.unique(first_ds_groups))))
    print("Second dataset number of unique storm days: " + str(len(np.unique(second_ds_groups))))
    print("First dataset total number of hail labels: " + str(np.sum(first_ds_labels["MESH_class_bin"].to_numpy())))
    print("Second dataset total number of hail labels: " + str(np.sum(second_ds_labels["MESH_class_bin"].to_numpy())))
    print("First dataset base rate: " + str(np.mean(first_ds_labels["MESH_class_bin"].to_numpy())))
    print("Second dataset base rate: " + str(np.mean(second_ds_labels["MESH_class_bin"].to_numpy())))


def create_parser():
    '''
    Create argument parser
    '''
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description='Data Splitting', fromfile_prefix_chars='@')

    parser.add_argument('--examples_glob', type=str, default="/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-1_hour-1_inch-train_val_test-cross_val/patches/trainvaltest/examples/*")
    parser.add_argument('--labels_glob', type=str, default="/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-1_hour-1_inch-train_val_test-cross_val/patches/trainvaltest/labels/*")
    parser.add_argument('--fold_path', type=str, default="/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour-more_fields-1_inch-cross_val/patches/cv_folds")
    parser.add_argument('--num_putput_files', type=int, default=30)
    parser.add_argument('--n_folds', type=int, default=5)

    return parser


if __name__ == "__main__":
    # Perform all the argument parsing
    parser = create_parser()
    args = parser.parse_args()
    args = vars(args)

    stratified_kfold_cross_val(args["examples_glob"], args["labels_glob"], args["fold_path"], args["num_output_files"], args["n_folds"])


