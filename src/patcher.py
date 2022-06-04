import xarray as xr
import pygrib
import numpy as np
import configparser
import os
import glob
from tqdm import tqdm
import re
import datetime
import copy 
import argparse
import warnings
import xesmf as xe


# NOTE: netcdf4 also need to be manually installed as dependencies
#       IMPORTANT: First thing to install on new env is xesmf/esmpy using the command conda install -c conda-forge xesmf esmpy=8.0.0
#       this will install python as well. Second thing to install is cftime. Third thing is to install dask. Then see netcdf4 above

# NOTE: Spatial dataset delineation in file/path naming not currently supported.


# TODO: Make documentation on how data directories should be organized. (Data downloader should make them this way)
#       They should be made such that the full date (but not necessarily the hour) is always either a directory 
#       (or a set of directories) somewhere in each file's path and/or be included in the name of each file.
#       It should note that it will fail if those .idx files are in the label/feature directories. Only expected files should be in there.
class Patcher:
    def __init__(self, config_path, run_num):
        # Parse in config file specified by config_path. See examples given in repo
        config = configparser.ConfigParser()
        config.read(config_path)
        self.config = cfg_parser(config)
        self.run_num = run_num

    
    # TODO: Move this to the IOHelper script with collection of functions
    # TODO: Fix this with changed create_file_list
    def list_possible_level_types(self, list_labels=False):
        # Pull first file in data directory to sample metadata
        all_data_files = self.create_file_list(list_labels = list_labels)
        one_file = all_data_files[0]

        # Level types must be printed in a different way depending on filetype
        if '.grib' in os.path.split(one_file)[-1]:
            grbs = pygrib.open(one_file)
            all_level_types = np.unique([grb["typeOfLevel"] for grb in grbs])
            print(all_level_types)
            grbs.close()

        elif '.nc' in os.path.split(one_file)[-1]:
            print("File is not of type \"grib\" or \"grib2\", netCDF not implemented yet.")
            # TODO: Implement
        
        else:
            raise IOError(os.path.split(one_file)[-1] + " is an unsupported filetype. Only filetypes supported currenly are .nc and .grib/.grib2")


    # TODO: Move this to the IOHelper script with collection of functions
    # TODO: Fix this with changed create_file_list
    def list_possible_root_keys(self, list_labels=False):
        # Pull first file in data directory to sample metadata
        all_data_files = self.create_file_list(list_labels = list_labels)
        one_file = all_data_files[0]

        # Keys must be printed in a different way depending on filetype
        if '.grib' in os.path.split(one_file)[-1]:
            grbs = pygrib.open(one_file)
            root_keys = np.unique(np.concatenate(np.array([grb.keys() for grb in grbs], dtype=object)))
            print(root_keys)
            grbs.close()

        elif '.nc' in os.path.split(one_file)[-1]:
            print("File is not of type \"grib\" or \"grib2\", netCDF not implemented yet.")
            # TODO: Implement
        
        else:
            raise IOError(os.path.split(one_file)[-1] + " is an unsupported filetype. Only filetypes supported currenly are .nc and .grib/.grib2")


    # TODO: MAJOR: Make sure this randomizes time order so it doesn't always return first time first
    # TODO: Check to see if I need to close xarray datasets for memory issues
    # NOTE: Only works in multi-dateset case. Not to be used for 1 dataset case.
    def _compare_datetimes_with_IO(self, current_index, chosen_date_indeces, all_datetimes,
                                         data_settings_cfgs, datasets_paths, chosen_resolution, 
                                         solution_indeces_files=None, solution_indeces_times=None, chosen_datetimes_adjusted=None):
        found_files = False
        if solution_indeces_files is None:
            solution_indeces_files = np.zeros(len(chosen_date_indeces))
        if solution_indeces_times is None:
            solution_indeces_times = np.zeros(len(chosen_date_indeces))
        if chosen_datetimes_adjusted is None:
            chosen_datetimes_adjusted = np.zeros(len(chosen_date_indeces))

        for i in range(solution_indeces_files[current_index], len(chosen_date_indeces[current_index])):
            if data_settings_cfgs[current_index]["Data"]["has_time_cord"]:
                time_cord_name = data_settings_cfgs[current_index]["Data"]["time_cord_name"]
                ds = xr.open_dataset(datasets_paths[current_index][chosen_date_indeces[current_index][i]])

                for j in range(solution_indeces_times[current_index], len(ds[time_cord_name])):
                    current_datetime = ds[time_cord_name][j].to_numpy().astype(chosen_resolution)
                    chosen_datetimes_adjusted[current_index] = current_datetime

                    if current_index == 0:
                        found_files, solution_indeces_files, solution_indeces_times = self._compare_datetimes_with_IO(current_index + 1, chosen_date_indeces, 
                                                                                                                    data_settings_cfgs, datasets_paths, 
                                                                                                                    chosen_resolution, solution_indeces_files, 
                                                                                                                    solution_indeces_times, chosen_datetimes_adjusted)

                    elif chosen_datetimes_adjusted[current_index-1] == current_datetime:
                        if current_index == len(chosen_date_indeces)-1:
                            found_files = True

                        else:
                            found_files, solution_indeces_files, solution_indeces_times = self._compare_datetimes_with_IO(current_index + 1, chosen_date_indeces, 
                                                                                                                        data_settings_cfgs, datasets_paths, 
                                                                                                                        chosen_resolution, solution_indeces_files, 
                                                                                                                        solution_indeces_times, chosen_datetimes_adjusted)

                    if found_files:
                        solution_indeces_times[current_index] = j
                        break

            else:
                current_datetime = all_datetimes[current_index][chosen_date_indeces[current_index][i]].astype(chosen_resolution)
                chosen_datetimes_adjusted[current_index] = current_datetime
                solution_indeces_times[current_index] = None

                if current_index == 0:
                    found_files, solution_indeces_files, solution_indeces_times = self._compare_datetimes_with_IO(current_index + 1, chosen_date_indeces, 
                                                                                                                    data_settings_cfgs, datasets_paths, 
                                                                                                                    chosen_resolution, solution_indeces_files, 
                                                                                                                    solution_indeces_times, chosen_datetimes_adjusted)

                elif chosen_datetimes_adjusted[current_index-1] == current_datetime:
                    if current_index == len(chosen_date_indeces)-1:
                        found_files = True

                    else:
                        found_files, solution_indeces_files, solution_indeces_times = self._compare_datetimes_with_IO(current_index + 1, chosen_date_indeces, 
                                                                                                                    data_settings_cfgs, datasets_paths, 
                                                                                                                    chosen_resolution, solution_indeces_files, 
                                                                                                                    solution_indeces_times, chosen_datetimes_adjusted)                                     

            if found_files:
                solution_indeces_files[current_index] = i
                break
            else:
                solution_indeces_times[current_index] = 0

        if not found_files:
            solution_indeces_files[current_index] = 0


        return found_files, solution_indeces_files, solution_indeces_times

    
    def run(self, settings_dict):
        patch_size = settings_dict["Patches"]["patch_size"]
        feature_patches_root = settings_dict["Output"]["examples_root"]
        label_patches_root = settings_dict["Output"]["labels_root"]
        n_patches = settings_dict["Patches"]["number_of_patches"]
        patches_per_time = settings_dict["Patches"]["patches_per_unit_time"]
        chosen_resolution = settings_dict["Patches"]["chosen_resolution"]

        data_settings_cfgs = []
        for data_settings_cfg in settings_dict["Input_Data"]["input_cfg_list"]:
            config = configparser.ConfigParser()
            config.read(data_settings_cfg)
            data_settings_cfgs.append(cfg_parser(config))

        # TODO: Maybe put the following loop into its own method?
        datasets_paths = []
        datasets_datetimes = []
        datasets_date_resolutions = []
        datasets_date_resolution_vals = []
        for data_settings_cfg in data_settings_cfgs:
            file_list = self.create_file_list(data_settings_cfg["Path"]["root_dir"], 
                                              data_settings_cfg["Path"]["path_glob"], 
                                              data_settings_cfg["Path"]["path_reg"])

            dateset_datetimes, dataset_date_resolution, dataset_date_resolution_val = self.extract_best_datetime_no_IO(data_settings_cfg["Path"]["root_dir"],
                                                                                      file_list, data_settings_cfg["Path"]["dt_positions"],
                                                                                      data_settings_cfg["Path"]["dt_regs"],
                                                                                      data_settings_cfg["Path"]["dt_formats"])

            file_list = self.select_data_range(file_list, data_settings_cfg["Bounds"]["data_start"], data_settings_cfg["Bounds"]["data_end"],
                                               data_settings_cfg["Bounds"]["use_date_for_data_range"], dateset_datetimes, dataset_date_resolution)

            datasets_paths.append(file_list)
            datasets_datetimes.append(dateset_datetimes)
            datasets_date_resolutions.append(dataset_date_resolution)
            datasets_date_resolution_vals.append(dataset_date_resolution_val)

        # Set top level lists to numpy arrays for more functionality
        datasets_paths = np.array(datasets_paths, dtype=list)
        datasets_datetimes = np.array(datasets_datetimes, dtype=list)
        datasets_date_resolutions = np.array(datasets_date_resolutions)
        datasets_date_resolution_vals = np.array(datasets_date_resolution_vals)

        # Sort by how high each dataset's resolution is
        inds = datasets_date_resolution_vals.argsort()
        datasets_paths = datasets_paths[inds]
        datasets_datetimes = datasets_datetimes[inds]
        datasets_date_resolutions = datasets_date_resolutions[inds]
        datasets_date_resolution_vals = datasets_date_resolution_vals[inds]

        loop_number = n_patches // patches_per_time
        feature_patches = None
        label_patches = None
        date_indeces = np.random.choice(np.arange(0,len(datasets_paths[-1])), size=len(datasets_paths[-1]), replace=False)
        date_counter = 0
        time_counter = 0
        time_indeces = None # Used if single dataset case with time. (for shuffling data)

        # Have numpy datetime64 objects with dates adjusted for the lowest resolution we have ready for all datasets
        # TODO: Double check this idea by considering it once more
        lowest_resolution_dates = []
        for dataset_datetimes in datasets_datetimes:
            datetimes_adjusted = []
            for dataset_datetime in dataset_datetimes:
                datetimes_adjusted.append(dataset_datetime.astype(datasets_date_resolutions[0]))
            lowest_resolution_dates.append(datetimes_adjusted)

        chosen_date_indeces = []
        for i in range(len(datasets_paths)):
            chosen_date_indeces.append([])

        # Main high level variables that govern the flow of patching.
        load_new_files = True
        solution_indeces_files = None
        solution_indeces_times = None
        found_files = False

        # Setup solution_indeces_files and solution_indeces_times for case where we only are loading one dataset
        if len(chosen_date_indeces) == 1:
            solution_indeces_files = date_indeces

        for n in tqdm(np.arange(0,loop_number)):

            if len(chosen_date_indeces) != 1:

                while not found_files:

                    while date_counter < len(date_indeces) and load_new_files:

                        for i, dataset_datetimes in enumerate(lowest_resolution_dates):

                            dataset_datetimes = np.array(dataset_datetimes) #TODO: check if I have to specify datetime64 here?
                            chosen_date_indeces[i] = np.where(dataset_datetimes == dataset_datetimes[-1][date_indeces[date_counter]])[0]

                        date_counter = date_counter + 1
                        if np.all([len(i) != 0 for i in chosen_date_indeces]):
                            load_new_files = False
                            solution_indeces_files = np.zeros(len(chosen_date_indeces))
                            solution_indeces_times = np.zeros(len(chosen_date_indeces))

                    if date_counter == len(date_indeces) and load_new_files:
                        break

                    found_files, solution_indeces_files, solution_indeces_times = self._compare_datetimes_with_IO(0,chosen_date_indeces, datasets_datetimes,
                                                                                                                data_settings_cfgs,datasets_paths,chosen_resolution,
                                                                                                                solution_indeces_files=solution_indeces_files,
                                                                                                                solution_indeces_times=solution_indeces_times)

                    if not found_files:
                        load_new_files = True

                if date_counter == len(date_indeces) and load_new_files:
                    warnings.warn('WARNING: Ran out of files with matching datetimes. Number of completed patches may be less than expected. Please consider adjusting "patches_per_unit_time"')
                    break

                found_files = False

            else:
                solution_indeces_files = [date_indeces[date_counter]]
                solution_indeces_times = [None]
                if data_settings_cfgs[0]["Data"]["has_time_cord"]:
                    if time_counter == 0:
                        ds = xr.open_dataset(datasets_paths[0][date_indeces[date_counter]])
                        time_cord_name = data_settings_cfgs[0]["Data"]["time_cord_name"]
                        current_datetime = ds[time_cord_name]

                        time_indeces = np.random.choice(np.arange(0,len(current_datetime)), size=len(current_datetime), replace=False)

                    solution_indeces_times = [time_indeces[time_counter]]

            loaded_datasets = []
            reproj_ds = None
            reproj_ds_index = 0
            for i, (file_index, time_index) in enumerate(zip(solution_indeces_files, solution_indeces_times)):
                ds = xr.open_dataset(datasets_paths[i][file_index])
                if data_settings_cfgs[i]["Data"]["has_time_cord"]:
                    time_dim_name = data_settings_cfgs[i]["Data"]["time_dim_name"]
                    ds = ds[{time_dim_name: time_index}]

                # Select only the data we want
                ds = ds[data_settings_cfgs[i]["Data"]["selected_vars"]]

                lons = ds[data_settings_cfgs[i]["Data"]["lon_cord_name"]].to_numpy()
                lats = ds[data_settings_cfgs[i]["Data"]["lat_cord_name"]].to_numpy()

                if len(lons.shape) == 1:
                    lons, lats = np.meshgrid(lons, lats)
                elif len(lons.shape) == 2:
                    pass
                else:
                    raise Exception("At least one dataset has lat/lons with either too few or too many dimensions. lat/lons must be either 2d or 1d.")

                ds = ds.assign_coords(lon=((data_settings_cfgs[i]["Data"]["y_dim_name"],data_settings_cfgs[i]["Data"]["x_dim_name"]), lons))
                ds = ds.assign_coords(lat=((data_settings_cfgs[i]["Data"]["y_dim_name"],data_settings_cfgs[i]["Data"]["x_dim_name"]), lats))

                if data_settings_cfgs[i]["Data"]["reproj_target"]:
                    reproj_ds = ds
                    reproj_ds_index = i

                loaded_datasets.append(ds)

            if reproj_ds is None:
                raise Exception("No dataset has been designated as the reproj_target. You must designate exactly one as this. Even if only loading one dataset.")

            # Increment index system for loop around
            if data_settings_cfgs[-1]["Data"]["has_time_cord"]:
                solution_indeces_times[-1] = solution_indeces_times[-1] + 1
            else:
                solution_indeces_files[-1] = solution_indeces_files[-1] + 1

            # TODO: Maybe make the feature_datasets and label_datasets here instead?
            reproj_datasets = []
            possible_x_indeces_for_patches = None
            possible_y_indeces_for_patches = None
            smallest_indeces_list_len = np.inf
            dataset_empty_or_out_of_range = False
            for ds in loaded_datasets:
                # TODO: Maybe make the regridder algorithm a config setting?
                # TODO: Consider if reuse_weights:bool is useful here
                regridder = xe.Regridder(ds, reproj_ds, "bilinear", unmapped_to_nan=True)
                ds_reproj = regridder(ds)
                for key_name in ds_reproj.keys():
                    data_var = ds_reproj[key_name].to_numpy()
                    if np.isnan(data_var).all():
                        dataset_empty_or_out_of_range = True
                        break

                if dataset_empty_or_out_of_range:
                    break

                # ds_reproj = ds_reproj.where(np.logical_not(np.isnan(ds_reproj)), drop=True)
                for key_name in ds_reproj.keys():
                    data_var = ds_reproj[key_name].to_numpy()
                    possible_pixels = np.where(np.logical_not(np.isnan(data_var)))
                    if len(possible_pixels[0]) < smallest_indeces_list_len:
                        smallest_indeces_list_len = len(possible_pixels[0])
                        possible_x_indeces_for_patches = possible_pixels[0]
                        possible_y_indeces_for_patches = possible_pixels[1]

                reproj_datasets.append(ds_reproj)

            if dataset_empty_or_out_of_range:
                warnings.warn('WARNING: At least one of the selected dataset files contained data that was entirely missing or data that did not spatially aligned with the other data. Had to skip and total number of patches may now be less than expected.')
                continue

            pixel_counter = 0
            pixel_indeces = np.random.choice(np.arange(0,len(possible_x_indeces_for_patches)), size=len(possible_x_indeces_for_patches), replace=False)
            x_max = reproj_ds.dims[data_settings_cfgs[reproj_ds_index]["Data"]["x_dim_name"]]
            y_max = reproj_ds.dims[data_settings_cfgs[reproj_ds_index]["Data"]["y_dim_name"]]
            patches = []
            grid_size = [x_max, y_max]
            for i in range(patches_per_time):
                has_nans_in_patch = True
                while has_nans_in_patch and pixel_counter < len(pixel_indeces):
                    has_nans_in_patch = False
                    x_i = possible_x_indeces_for_patches[pixel_indeces[pixel_counter]]
                    y_i = possible_y_indeces_for_patches[pixel_indeces[pixel_counter]]
                    for ds in reproj_datasets:
                        patch = self._make_patch(ds, grid_size, patch_size, x_i, y_i)
                        for key_name in patch.keys():
                            data_var = patch[key_name].to_numpy()
                            if np.isnan(data_var).any():
                                has_nans_in_patch = True
                        if has_nans_in_patch:
                            patches = []
                            break
                        patches.append(patch)
                    pixel_counter = pixel_counter + 1

                for j, patch in enumerate(patches):
                    if data_settings_cfgs[j]["Data"]["is_label_data"]:
                        label_patches = self._concat_patches(label_patches, patch)
                    else:
                        feature_patches = self._concat_patches(feature_patches, patch)
            
        if feature_patches is not None:
            feature_patch_path = os.path.join(feature_patches_root, str(self.run_num) + ".nc")
            feature_patches.to_netcdf(feature_patch_path)
        
        if label_patches is not None:
            label_patch_path = os.path.join(label_patches_root, str(self.run_num) + ".nc")
            label_patches.to_netcdf(label_patch_path)

    
    def _concat_patches(self, patches, patch):
        if patches is None:
            patches = copy.deepcopy(patch.expand_dims(dim='n_samples'))
        else:
            patches = xr.concat([patches,patch.expand_dims(dim='n_samples')],dim='n_samples')

        return patches
    

    def _make_patch(self, file_ds, grid_size, patch_size, x, y):
        halfstep = int(patch_size/2)
        y_max = grid_size[1]-halfstep
        x_max = grid_size[0]-halfstep

        patch = None

        if x >= halfstep and x < x_max and y >= halfstep and y < y_max:
            patch = file_ds.isel(x=slice(x-halfstep,x+halfstep),y=slice(y-halfstep,y+halfstep))

        return patch
    

    # # Also removes any feature file that does not have a corresponding label
    # def _get_label_files(self, feature_files):
    #     feature_files_to_remove = []
    #     label_files = []
    #     for feature_file in feature_files:
    #         label_file = self._find_label_path(feature_file)
    #         if label_file is None:
    #             feature_files_to_remove.append(feature_file)
    #         else:
    #             label_files.append(label_file)
            
    #     for file_to_remove in feature_files_to_remove:
    #         feature_files.remove(file_to_remove)

    #     if len(feature_files) != len(label_files):
    #         raise Exception("Method _get_label_files has failed. Not all dud feature files removed.")

    #     return feature_files, label_files


    # Three possible options:
    # 1. use dates to set range
    # 2. use hard indeces to set range
    # 3. use percentages (real numbers) to set range
    # NOTE: Does not check validity of selections. May add that later
    # NOTE: Please make sure the start and end dates in ISO_8601 format
    def select_data_range(self, file_list, data_start, data_end, use_date=False, dates_list=None, dataset_date_resolution=None):
        file_list = np.array(file_list)

        if use_date:
            if data_start is not None:
                start_date = np.datetime64(data_start).astype(dataset_date_resolution)
            if data_end is not None:
                end_date = np.datetime64(data_end).astype(dataset_date_resolution)

            if data_start is not None and data_end is not None:
                return file_list[np.where(np.logical_and(dates_list >= start_date, dates_list <= end_date))].tolist()
            elif data_start is None and data_end is not None:
                return file_list[np.where(dates_list <= end_date)].tolist()
            elif data_start is not None and data_end is None:
                return file_list[np.where(dates_list >= start_date)].tolist()
            else:
                return file_list.tolist()
        
        else:
            start_index = 0
            end_index = -1

            if data_start is not None:
                if isinstance(data_start, int):
                    start_index = data_start
                elif isinstance(data_start, float):
                    start_index = int(data_start*len(file_list))

            if data_end is not None:
                if isinstance(data_end, int):
                    end_index = data_end
                elif isinstance(data_end, float):
                    end_index = int(data_end*len(file_list))
        
            return file_list[start_index:end_index].tolist()
                

    # def _find_label_path(self, feature_file):
    #     feature_root = self.config["Input"]["examples_root"]
    #     feature_glob = self.config["Input"]["examples_glob"]
    #     feature_hour_regex = self.config["Input"]["examples_hour_regex"]
    #     feature_minute_regex = self.config["Input"]["examples_minute_regex"]
    #     labels_root = self.config["Input"]["labels_root"]
    #     labels_glob = self.config["Input"]["labels_glob"]
    #     labels_hour_regex = self.config["Input"]["labels_hour_regex"]
    #     labels_minute_regex = self.config["Input"]["labels_minute_regex"]

    #     glob_path = self._glob_path_maker(feature_root, feature_glob)
    #     feature_path_date_index = self._date_dir_index_finder(glob_path)
    #     feature_path_array = feature_file.split("/")
    #     date = feature_path_array[feature_path_date_index]

    #     glob_path = self._glob_path_maker(labels_root, labels_glob)
    #     label_path_date_index = self._date_dir_index_finder(glob_path)
    #     glob_path_array = glob_path.split("/")
    #     glob_path_array[label_path_date_index] = date

    #     feature_filename = os.path.split(feature_file)[-1]
    #     feature_time = self._get_time_from_regex(feature_hour_regex, feature_minute_regex, feature_filename)

    #     label_glob_with_date = "/".join(glob_path_array)
    #     label_files = glob.glob(label_glob_with_date)

    #     label_path = None
    #     for label_file in label_files:
    #         label_filename = os.path.split(label_file)[-1]
    #         label_time = self._get_time_from_regex(labels_hour_regex, labels_minute_regex, label_filename)

    #         if feature_time == label_time:
    #             label_path = label_file
    #             break

    #     return label_path


    # def _date_dir_index_finder(self, glob_path):
    #     glob_path_array = np.array(glob_path.split("/"))
    #     date_index = np.where(glob_path_array == "*")[0][0] # WAS date_index = np.where(glob_path_array == "*")[0][0]
    #     return date_index


    # NOTE: Hour MUST be in format of %H%H and minute MUST be in format of %M%M
    # def _get_time_from_regex(self, hour_reg, min_reg, filename):
    #     reg_hour = re.search(hour_reg, filename)
    #     if not reg_hour:
    #         raise IOError("A filename_hour_regex is not valid. Can't find hour.")
    #     hour = reg_hour.group(1)

    #     if min_reg is not None:
    #         reg_min = re.search(min_reg, filename)
    #         if not reg_min:
    #             raise IOError("A filename_minute_regex is not valid. Can't find minute.")
    #         min = reg_min.group(1)
    #     else:
    #         min = "00"
        
    #     time = hour + min
    #     return time


    # Extracts the best possible time information from data's directory structure and file name 
    # NOTE: See the required characters for designating where each datetime component is located in each file's path
    # NOTE: If a lower level datetime unit is used (for example hours or minute), every higher level must also be present
    # NOTE: datetime_positions chars differ from the datetime.datetime chars needed for "datetime_ISO_formats". May change this later
    # NOTE: In the path/name of each file there must be at least SOME datetime information. The no-information scenario is not currently supported
    def extract_best_datetime_no_IO(self, root_path, data_file_list, datetime_positions, extraction_regs, datetime_ISO_formats):
        datetime_positions_path = root_path.rstrip("/") + datetime_positions
        datetime_chars = datetime_positions_path.split("/")

        datetime_chars_indeces = []
        datetime_chars_seperated = []
        for i, datetime_str in enumerate(datetime_chars):
            for datetime_char in datetime_str:
                if datetime_char not in ["Y","M","D","h","m"]:
                    raise Exception("A given datetime position character is not one of the accepted options.")
                datetime_chars_seperated.append(datetime_char)
                datetime_chars_indeces.append(i)

        if len(datetime_chars_indeces) == 0:
            raise Exception("Giving no datetime information in the file names or in each file's path is not supported.")

        # TODO: Double check datetime64[char] have been chosen right
        if "m" in datetime_chars_seperated:
            dataset_date_resolution = "datetime64[m]"
            dataset_date_resolution_val = 4
        elif "h" in datetime_chars_seperated:
            dataset_date_resolution = "datetime64[h]"
            dataset_date_resolution_val = 3
        elif "D" in datetime_chars_seperated:
            dataset_date_resolution = "datetime64[D]"
            dataset_date_resolution_val = 2
        elif "M" in datetime_chars_seperated:
            dataset_date_resolution = "datetime64[M]"
            dataset_date_resolution_val = 1
        else:
            dataset_date_resolution = "datetime64[Y]"
            dataset_date_resolution_val = 0

        files_to_remove = []
        datetimes = []
        for data_file in data_file_list:
            file_datetime_read_fail = False
            data_file_split = data_file.split("/")

            minute_str = ""
            hour_str = ""
            day_str = ""
            month_str = ""
            year_str = ""
            
            for i, datetime_char in enumerate(datetime_chars_seperated):
                reg_extracted = re.search(extraction_regs[i], data_file_split[datetime_chars_indeces[i]])
                if not reg_extracted:
                    file_datetime_read_fail = True
                    break
                time_component_str = reg_extracted.group(1)

                if datetime_char == "Y":
                    datetime_obj = datetime.strptime(time_component_str, datetime_ISO_formats[i])
                    year_str = datetime_obj.strftime("%Y")
                elif datetime_char == "M":
                    datetime_obj = datetime.strptime(time_component_str, datetime_ISO_formats[i])
                    month_str = "-" + datetime_obj.strftime("%m")
                elif datetime_char == "D":
                    datetime_obj = datetime.strptime(time_component_str, datetime_ISO_formats[i])
                    day_str = "-" + datetime_obj.strftime("%d")
                elif datetime_char == "h":
                    datetime_obj = datetime.strptime(time_component_str, datetime_ISO_formats[i])
                    hour_str = "T" + datetime_obj.strftime("%H")
                elif datetime_char == "m":
                    datetime_obj = datetime.strptime(time_component_str, datetime_ISO_formats[i])
                    minute_str = ":" + datetime_obj.strftime("%M")

            if file_datetime_read_fail:
                files_to_remove.append(data_file)
                continue

            datetime_np_str = year_str + month_str + day_str + hour_str + minute_str

            try:
                datetime_np = np.datetime64(datetime_np_str)
            except:
                files_to_remove.append(data_file)
                continue

            datetimes.append(datetime_np)

        return datetimes, dataset_date_resolution, dataset_date_resolution_val


    # TODO: May not need this in a function anymore???
    def _glob_path_maker(self, root_path, glob_string):
        return root_path.rstrip("/") + "/" + glob_string.lstrip("/")


    # NOTE: In path_glob only include wild card operators for each directory level you want to search across.
    #       The regex can handle any filtering.
    def create_file_list(self, root_dir, path_glob, path_regex):
        glob_path = self._glob_path_maker(root_dir, path_glob)
        unfiltered_file_list = glob.glob(glob_path)

        file_list = []
        wildcard_indeces = np.where(np.array(glob_path.split("/")) == "*")[0]
        for file in unfiltered_file_list:
            file_array = np.array(file.split("/"))
            reg_bools = []
            for i, reg in enumerate(path_regex):
                reg_bool = re.search(reg, file_array[wildcard_indeces[i]])
                reg_bools.append(reg_bool)

            if np.all(np.array(reg_bools)):
                file_list.append(file)

        file_list.sort()
        return file_list

        # if list_labels:
        #     root = self.config["Input"]["labels_root"]
        #     glob_string = self.config["Input"]["labels_glob"]
        # else:
        #     root = self.config["Input"]["examples_root"]
        #     glob_string = self.config["Input"]["examples_glob"]

        
        # return glob.glob(glob_path)


'''
Next two functions functions I wrote while at UBC. Useful but maybe could be updated.
'''

# NOTE: This method assumes that the config will never have things nested
# deeper than one level. This should always be the case in the .cfg anyways
def cfg_parser(cfg_object):
    new_cfg = {}
    cfg = cfg_object._sections
    for key_i, value_i in cfg.items():
        if type(value_i) is dict:
            for key_j, value_j in value_i.items():
                if key_i not in new_cfg:
                    new_cfg[key_i] = {}
                new_cfg[key_i][key_j] = value_parser(value_j)
        else:
            new_cfg[key_i] = value_parser(value_i)
    
    return new_cfg


# Not my favorite solution, but it is easy to read and understand.
# Should be robust as well.
def value_parser(value):
    try:
        return int(value)
    except:
        pass
    try:
        return float(value)
    except:
        pass
    if value.lower() == "false":
        return False
    if value.lower() == "true":
        return True
    if value.lower() == "none":
        return None
    value = value.replace("\"","")
    value = value.replace("'","")
    if value[0] == "[" and value[-1] == "]":
        output_value = []
        value = value.replace("[","")
        value = value.replace("]","")
        value = value.split(",")
        for i in value:
            output_value.append(value_parser(i.strip()))
        return output_value
    return value


if __name__ == "__main__":
         # Parse the command-line arguments
    parser = argparse.ArgumentParser(description='NetCDF Patch Generator')
    parser.add_argument('--run_num', type=int, help='Number to label this run')
    parser.add_argument('--config_path', type=str, help='Path to config file')
    args = parser.parse_args()


    # TODO: Switch this to command line argument
    patcher = Patcher(args.config_path, args.run_num)
    patcher.run()