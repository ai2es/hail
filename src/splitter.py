import numpy as np
import os 
import glob
import xarray as xr
from sklearn.cluster import DBSCAN
from sklearn.model_selection import StratifiedGroupKFold


SECOND_DS_SIZE = 0.5
EXAMPLES_GLOB = "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour-more_fields-1_inch-balanced/patches/valtest/examples/*"
LABELS_GLOB = "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour-more_fields-1_inch-balanced/patches/valtest/labels/*"
FIRST_DS_OUTPUT = "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour-more_fields-1_inch-balanced/patches/val_split"
SECOND_DS_OUTPUT = "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour-more_fields-1_inch-balanced/patches/test_split"
NUM_OUTPUT_FILES = 15


def cluster_by_storm_event(labels):
    labels = labels[{"time_dim": 0}] # Select only the first time (init time)

    sample_init_times = labels["time"].to_numpy()
    sample_init_times = sample_init_times.reshape(-1,1) # To fit (n_samples,1) convention needed for DBSCAN

    clustering = DBSCAN(eps=2.16e13, min_samples=1).fit(sample_init_times)

    return clustering.labels_


def get_group_positions(labels, storm_groups, second_ds_size):
    labels = labels["MESH_class_bin"].drop(["time","lon","lat"]).to_numpy()

    for i in np.arange(1,len(labels.shape)):
        storm_groups = np.expand_dims(storm_groups, -1)
    storm_groups = np.tile(storm_groups, labels.shape[1:])

    labels = labels.ravel()
    storm_groups = storm_groups.ravel()
    fake_X = np.zeros(len(labels))

    sgkf = StratifiedGroupKFold(n_splits=int(1/second_ds_size))

    first_ds_indices, second_ds_indices = sgkf.split(fake_X, labels, storm_groups)

    # 0 index for one "fold"
    first_ds_indices = first_ds_indices[0]
    second_ds_indices = second_ds_indices[0]

    return np.unique(storm_groups[first_ds_indices]), np.unique(storm_groups[second_ds_indices])


def stratify_datasets_by_storm_events(examples_glob, labels_glob, second_ds_size, first_ds_output_path, second_ds_output_path, num_output_files):
    file_list = glob.glob(examples_glob)
    file_list.sort()
    examples_ds = xr.open_mfdataset(file_list, concat_dim='n_samples', combine='nested', engine='netcdf4')

    file_list = glob.glob(labels_glob)
    file_list.sort()
    label_ds = xr.open_mfdataset(file_list, concat_dim='n_samples', combine='nested', engine='netcdf4')

    storm_event_clusters = cluster_by_storm_event(label_ds)
    groups_in_first_ds, groups_in_second_ds = get_group_positions(label_ds,storm_event_clusters,second_ds_size)

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

    divded_indices_for_first_dataset_netcdfs = np.array_split(np.arange(first_example_ds.dims["n_samples"]), num_output_files)
    divded_indices_for_second_dataset_netcdfs = np.array_split(np.arange(second_example_ds.dims["n_samples"]), num_output_files)

    for i, (first_indices, second_indices) in enumerate(zip(divded_indices_for_first_dataset_netcdfs, divded_indices_for_second_dataset_netcdfs)):
        file_name_num = "{:04d}".format(i)

        first_example_ds[{"n_samples": first_indices}].to_netcdf(os.path.join(first_ds_output_path, "examples/" + file_name_num + ".nc"))
        first_labels_ds[{"n_samples": first_indices}].to_netcdf(os.path.join(first_ds_output_path, "labels/" + file_name_num + ".nc"))
        second_example_ds[{"n_samples": second_indices}].to_netcdf(os.path.join(second_ds_output_path, "examples/" + file_name_num + ".nc"))
        second_labels_ds[{"n_samples": second_indices}].to_netcdf(os.path.join(second_ds_output_path, "labels/" + file_name_num + ".nc"))


stratify_datasets_by_storm_events(EXAMPLES_GLOB, LABELS_GLOB, SECOND_DS_SIZE, FIRST_DS_OUTPUT, SECOND_DS_OUTPUT, NUM_OUTPUT_FILES)