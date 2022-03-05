from patcher import Patcher
from tqdm import tqdm
import numpy as np
import xarray as xr
import os
import argparse


class BalancedPatcher(Patcher):
    # NOTE: Currently only supports classification. Further inheritance required?
    def run(self):
        patch_size = self.config["Patches"]["patch_size"]
        feature_patches_root = self.config["Output"]["examples_root"]
        label_patches_root = self.config["Output"]["labels_root"]
        n_patches = self.config["Patches"]["number_of_patches"]
        all_labels = self.config["Patches"]["all_labels"]
        chosen_label = self.config["Patches"]["chosen_label"]
        center_shift = self.config["Patches"]["center_shift"]

        feature_files = self._glob_files()
        feature_files.sort()

        # Choose which files to make into patches
        feature_files = self._select_data_range(feature_files)
        feature_files, label_files = self._get_label_files(feature_files)

        # Find which files have each class present for class balancing purposes
        # Saved into memory in following loop block to avoid too much disk time
        classes_in_files = []
        for label_file in tqdm(label_files):
            label_file_ds = xr.open_dataset(label_file)
            label_file_ds.close()
            labels_np = label_file_ds[chosen_label].to_numpy()
            classes_in_file = []
            for label in all_labels:
                class_bool = label in np.unique(labels_np)
                classes_in_file.append(class_bool)
            classes_in_files.append(classes_in_file)

        n_patch_label_sets = int(n_patches/len(all_labels))

        feature_patches = None
        label_patches = None

        # TODO: Make this into open_mfdataset for proper parallel
        for n in tqdm(np.arange(0,n_patch_label_sets)):
            j = 0
            for label in all_labels:
                shuffled_indeces = np.random.choice(np.arange(0,len(feature_files)), size=len(feature_files), replace=False)

                for i in shuffled_indeces:
                    if classes_in_files[i][j]:
                        shuffled_feature_file = feature_files[i]
                        shuffled_label_file = label_files[i]

                        feature_file_ds = xr.open_dataset(shuffled_feature_file)
                        label_file_ds = xr.open_dataset(shuffled_label_file)
                        feature_file_ds.close()
                        label_file_ds.close()

                        labels_np = label_file_ds[chosen_label].to_numpy()
                        label_location_indeces = np.where(labels_np == label)
                        label_location_indeces_row = label_location_indeces[0]
                        label_location_indeces_col = label_location_indeces[1]

                        chosen_location = np.random.randint(0,len(label_location_indeces_row))

                        x_i = label_location_indeces_row[chosen_location] + np.random.randint(-center_shift,center_shift)
                        y_i = label_location_indeces_col[chosen_location] + np.random.randint(-center_shift,center_shift)

                        x_max = feature_file_ds.dims["x"]
                        y_max = feature_file_ds.dims["y"]
                        grid_size = [x_max, y_max]

                        # Next bit of code is taken from lydia's scripts with modifications
                        feature_patch, label_patch = self._make_patch(feature_file_ds, label_file_ds, grid_size, patch_size, x_i, y_i)

                        if feature_patch is None:
                            continue

                        # TODO: MAJOR: (SHOULD BE METHOD IN PARENT CLASS) Add system for selecting both feature variables and
                        # label variables from list of strings in config file. Single label currrently called "chosen_label"
                        # can still be used for class balance purposes but there may be interest in multiple labels down the road
                        # for regression. (Perhaps then the label choice system should in another child patcher?)
                        # TODO: Make sure it can also take time as a variable

                        feature_patches = self._concat_patches(feature_patches, feature_patch)
                        label_patches = self._concat_patches(label_patches, label_patch)

                        break

                j = j + 1

        if feature_patches is not None:
            feature_patch_path = os.path.join(feature_patches_root, str(self.run_num) + ".nc")
            label_patch_path = os.path.join(label_patches_root, str(self.run_num) + ".nc")

            feature_patches.to_netcdf(feature_patch_path)
            label_patches.to_netcdf(label_patch_path)


if __name__ == "__main__":
     # Parse the command-line arguments
    parser = argparse.ArgumentParser(description='NetCDF Patch Generator')
    parser.add_argument('--run_num', type=int, help='Number to label this run')
    parser.add_argument('--config_path', type=str, help='Path to config file')
    args = parser.parse_args()


    # TODO: Switch this to command line argument
    patcher = BalancedPatcher(args.config_path, args.run_num)
    patcher.run()