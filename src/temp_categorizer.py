import xarray as xr
from tensorflow import keras
import glob
import os

input_dir = "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/2022_07_26_run/patches/labels/*"
output_dir = "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/2022_07_26_run/patches/labels_fixed"

input_files = glob.glob(input_dir)
input_files.sort()

n_classes = 2

for input_file in input_files:
    input_file_array = input_file.split("/")
    file_name = input_file_array[-1]
    ds = xr.open_dataset(input_file, decode_cf=True)

    # STUFF TO ADD ###########################
    if n_classes >= 2:
        all_keys = [key for key in ds.keys()]
        previous_key = all_keys[-1]
        new_var = ds[previous_key]
        new_var_dim_names = new_var.dims

        new_var_cat = keras.utils.to_categorical(new_var, num_classes=n_classes)

        for i in range(new_var_cat.shape[-1]):
            new_var_class_i = new_var_cat[...,i]
            new_var_name_i = previous_key + "_" + str(i)
            ds = ds.assign({new_var_name_i: (new_var_dim_names, new_var_class_i)})
        
        ds = ds.drop(previous_key)
    #################################################

    output_path = os.path.join(output_dir, file_name)
    ds.to_netcdf(output_path)