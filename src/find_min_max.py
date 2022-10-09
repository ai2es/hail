import xarray as xr
import numpy as np
import glob
import os

EXAMPLES_NETCDF_GLOB = "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/trained_at_init_time/patches/train/examples/*"
LABELS_NETCDF_GLOB = "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/trained_at_init_time/patches/train/labels/*"
MIN_MAX_FILES_DIR = "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/trained_at_init_time/patches/mins_maxs"

examples_files = glob.glob(EXAMPLES_NETCDF_GLOB)
examples_files.sort()
labels_files = glob.glob(LABELS_NETCDF_GLOB)
labels_files.sort()

one_examples_ds = xr.open_dataset(examples_files[0], decode_cf=False)
example_keys = [key for key in one_examples_ds.keys()]
one_examples_ds.close()

one_labels_ds = xr.open_dataset(labels_files[0], decode_cf=False)
label_keys = [key for key in one_labels_ds.keys()]
one_labels_ds.close()

examples_max = np.full(len(example_keys), -np.inf)
labels_max = np.full(len(label_keys), -np.inf)
examples_min = np.full(len(example_keys), np.inf)
labels_min = np.full(len(label_keys), np.inf)

for j, (examples_file, labels_file) in enumerate(zip(examples_files, labels_files)):
    examples_ds = xr.open_dataset(examples_file, decode_cf=False)
    labels_ds = xr.open_dataset(labels_file, decode_cf=False)

    examples_array = examples_ds.to_array().to_numpy()
    labels_array = labels_ds.to_array().to_numpy()

    examples_ds.close()
    labels_ds.close()

    for i in range(len(examples_max)):
        examples_max[i] = np.max((examples_max[i], np.max(examples_array[i,...])))
        examples_min[i] = np.min((examples_min[i], np.min(examples_array[i,...])))

    for i in range(len(labels_max)):
        labels_max[i] = np.max((labels_max[i], np.max(labels_array[i,...])))
        labels_min[i] = np.min((labels_min[i], np.min(labels_array[i,...])))

    print("Finished file num: " + str(j))

examples_max_ds = xr.Dataset(data_vars=dict(zip(example_keys, examples_max)))
examples_min_ds = xr.Dataset(data_vars=dict(zip(example_keys, examples_min)))
labels_max_ds = xr.Dataset(data_vars=dict(zip(label_keys, labels_max)))
labels_min_ds = xr.Dataset(data_vars=dict(zip(label_keys, labels_min)))

examples_max_ds.to_netcdf(os.path.join(MIN_MAX_FILES_DIR, "examples_max.nc"))
examples_min_ds.to_netcdf(os.path.join(MIN_MAX_FILES_DIR, "examples_min.nc"))
labels_max_ds.to_netcdf(os.path.join(MIN_MAX_FILES_DIR, "labels_max.nc"))
labels_min_ds.to_netcdf(os.path.join(MIN_MAX_FILES_DIR, "labels_min.nc"))
