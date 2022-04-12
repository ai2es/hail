from patcher import Patcher
from tqdm import tqdm
import numpy as np
import xarray as xr
import os
import argparse


class SelectionPatcher(Patcher):
    def run(self):
        patch_size = self.config["Patches"]["patch_size"]
        feature_patches_root = self.config["Output"]["examples_root"]
        label_patches_root = self.config["Output"]["labels_root"]
        feature_file = self.config["Input"]["feature_file"]
        label_file = self.config["Input"]["label_file"]
        # lats = self.config["Patches"]["lats"]
        # lons = self.config["Patches"]["lons"]

        feature_file_ds = xr.open_dataset(feature_file)
        label_file_ds = xr.open_dataset(label_file)

        x_max = feature_file_ds.dims["x"]
        y_max = feature_file_ds.dims["y"]
        grid_size = [x_max, y_max]

        # TODO: Change from this temp hardcoded solution
        # Make system for taking lat lons above and use nearest points as x_i and y_i for all
        x_i = 416
        y_i = 906

        feature_patch, label_patch = self._make_patch(feature_file_ds, label_file_ds, grid_size, patch_size, x_i, y_i)

        feature_patch_path = os.path.join(feature_patches_root, str(self.run_num) + ".nc")
        label_patch_path = os.path.join(label_patches_root, str(self.run_num) + ".nc")
        feature_patch.to_netcdf(feature_patch_path)
        label_patch.to_netcdf(label_patch_path)

    
if __name__ == "__main__":
     # Parse the command-line arguments
    parser = argparse.ArgumentParser(description='NetCDF Patch Generator')
    parser.add_argument('--run_num', type=int, help='Number to label this run')
    parser.add_argument('--config_path', type=str, help='Path to config file')
    args = parser.parse_args()


    # TODO: Switch this to command line argument
    patcher = SelectionPatcher(args.config_path, args.run_num)
    patcher.run()