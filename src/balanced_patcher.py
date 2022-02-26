from patcher import Patcher
from tqdm import tqdm
import numpy as np
import xarray as xr
import os


# TODO: Plot this output before continueing
class BalancedPatcher(Patcher):
    # NOTE: Currently only supports classification. Further inheritance required?
    def run(self):
        patch_size = self.config["Patches"]["patch_size"]
        feature_patches_root = self.config["Output"]["examples_root"]
        label_patches_root = self.config["Output"]["labels_root"]
        n_patches = self.config["Patches"]["number_of_patches"]
        all_labels = self.config["Patches"]["all_labels"]

        feature_files = self._glob_files()
        feature_files.sort()

        # Choose which files to make into patches
        feature_files = self._select_data_range(feature_files)

        n_patch_label_sets = int(n_patches/len(all_labels))

        for n in tqdm(np.arange(0,n_patch_label_sets)):
            for label in all_labels:
                shuffled_feature_files = np.random.choice(feature_files, size=len(feature_files), replace=False)
                shuffled_feature_files, shuffled_label_files = self._get_label_files(shuffled_feature_files)

                # feature_patch = None
                # label_patch = None

                for i in np.arange(0,len(shuffled_feature_files)):
                    shuffled_label_file = shuffled_label_files[i]
                    label_file_ds = xr.open_dataset(shuffled_label_file)

                    # TODO: SWITCH "mesh" TO CONFIG FILE LABEL NAME SELECTION
                    labels_np = label_file_ds.mesh.to_numpy()

                    if label in np.unique(labels_np):
                        shuffled_feature_file = shuffled_feature_files[i]
                        feature_file_ds = xr.open_dataset(shuffled_feature_file)

                        label_location_indeces = np.where(labels_np == label)
                        label_location_indeces_row = label_location_indeces[0]
                        label_location_indeces_col = label_location_indeces[1]

                        chosen_location = np.random.randint(0,len(label_location_indeces_row))

                        # TODO: Make sure these values get shifted so it is not always in the center of the patch
                        x_i = label_location_indeces_row[chosen_location]
                        y_i = label_location_indeces_col[chosen_location]

                        x_max = feature_file_ds.dims["x"]
                        y_max = feature_file_ds.dims["y"]
                        grid_size = [x_max, y_max]

                        # Next bit of code is taken from lydia's scripts with modifications
                        feature_patch, label_patch = self._make_patch(feature_file_ds, label_file_ds, grid_size, patch_size, x_i, y_i)

                        if feature_patch is None:
                            continue

                        feature_patch_path = os.path.join(feature_patches_root, str(n) + "-" + str(label) + ".nc")
                        label_patch_path = os.path.join(label_patches_root, str(n) + "-" + str(label) + ".nc")

                        feature_patch.to_netcdf(feature_patch_path)
                        label_patch.to_netcdf(label_patch_path)

                        break


if __name__ == "__main__":
    patcher = BalancedPatcher("/Users/tschmidt/Desktop/Hail_Nowcasting/balanced_patcher.cfg")
    patcher.run()