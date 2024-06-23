import numpy as np
import os
import json
import xarray as xr
import glob


EXAMPLES_GLOB = "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/unprocessed_refl/examples/*"
LABELS_GLOB = "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/unprocessed_refl/labels/*"
JSON_PATH = "/home/tgschmidt/hail/configs/slicing.json"
MODELS_ROOT_PATH = "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/multi_models"
NUM_OUTPUT_FILES = 10


def slice_by_indices(ds, slice_and_name_dict_list, root_output_path, num_output_files):
    all_var_names_used = []
    for slice_and_name_dict in slice_and_name_dict_list:
        var_names = slice_and_name_dict["data_var_names"]
        all_var_names_used.extend(var_names)
    new_ds = ds.drop(all_var_names_used)
    
    for slice_and_name_dict in slice_and_name_dict_list:
        var_names = slice_and_name_dict["data_var_names"]
        indices = slice_and_name_dict["slices"]

        if type(var_names) is not list or type(indices) is not list:
            raise Exception("Variable names must be made up of list of lists. Same with the slice indices.")
        
        ds_sliced = ds[var_names]
        ds_sliced = ds_sliced[{"time_dim": indices}]

        for i in range(ds_sliced.dims["time_dim"]):
            ds_one_timestep = ds_sliced[{"time_dim": i}]
            for data_var_name in ds_one_timestep.keys():
                data_var_time_name = data_var_name + "_" + str(i)
                new_ds = new_ds.assign({data_var_time_name: ds_one_timestep[data_var_name]})

    divided_indices_for_output = np.array_split(np.arange(new_ds.dims["n_samples"]), num_output_files)
    for i, output_indices in enumerate(divided_indices_for_output):
        file_name_num = "{:04d}".format(i)
        new_ds[{"n_samples": output_indices}].to_netcdf(os.path.join(root_output_path, file_name_num + ".nc"))


def slice_all_datasets(examples_glob, labels_glob, json_path, models_root_path, num_output_files):
    file_list = glob.glob(examples_glob)
    file_list.sort()
    examples_ds = xr.open_mfdataset(file_list, concat_dim='n_samples', combine='nested', engine='netcdf4')

    file_list = glob.glob(labels_glob)
    file_list.sort()
    label_ds = xr.open_mfdataset(file_list, concat_dim='n_samples', combine='nested', engine='netcdf4')

    with open(json_path) as json_file:
        slices_dict = json.load(json_file)

    slices_list = slices_dict["slices_settings"]

    for i, model_slices in enumerate(slices_list):
        model_str_num = "{:04d}".format(i)
        model_file_path = os.path.join(models_root_path, "2d_model_" + model_str_num)
        examples_file_path = os.path.join(model_file_path, "trainval/examples")
        labels_file_path = os.path.join(model_file_path, "trainval/labels")

        examples_settings_list = model_slices["examples"]
        labels_settings_list = model_slices["labels"]
        
        slice_by_indices(examples_ds, examples_settings_list, examples_file_path, num_output_files)
        slice_by_indices(label_ds, labels_settings_list, labels_file_path, num_output_files)


slice_all_datasets(EXAMPLES_GLOB, LABELS_GLOB, JSON_PATH, MODELS_ROOT_PATH, NUM_OUTPUT_FILES)