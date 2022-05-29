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


# NOTE: netcdf4 also need to be manually installed as dependencies
#       IMPORTANT: First thing to install on new env is xesmf/esmpy using the command conda install -c conda-forge xesmf esmpy=8.0.0
#       this will install python as well. Second thing to install is cftime. Third thing is to install dask. Then see netcdf4 above


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

    
    def run(self):
        patch_size = self.config["Patches"]["patch_size"]
        feature_patches_root = self.config["Output"]["examples_root"]
        label_patches_root = self.config["Output"]["labels_root"]
        n_patches = self.config["Patches"]["number_of_patches"]

        feature_files = self.create_file_list()
        feature_files.sort()

        # Choose which files to make into patches
        feature_files = self._select_data_range(feature_files)

        # Get a label file for each feature file and
        # remove any feature files that don't have a corresponding label
        # TODO: Make boolean check system for case where we don't want to make label patches at the same time as feature patches
        # BUT PATCHES WILL STILL BE NEEDED FOR CHECKING IF FEATURE FILES NEED TO BE REMOVED
        feature_files, label_files = self._get_label_files(feature_files)

        feature_patches = None
        label_patches = None

        for n in tqdm(np.arange(0,n_patches)):
            date_index = np.random.randint(0,len(feature_files))
            feature_file = feature_files[date_index]
            label_file = label_files[date_index]

            # NOTE: netcdf4 package required
            #       assumes that feature and label gridsizes are the same. DataMiner should enforce this.
            feature_file_ds = xr.open_dataset(feature_file)
            label_file_ds = xr.open_dataset(label_file)

            x_max = feature_file_ds.dims["x"]
            y_max = feature_file_ds.dims["y"]
            grid_size = [x_max, y_max]

            x_i = np.random.randint(0,x_max)
            y_i = np.random.randint(0,y_max)

            # Next bit of code is taken from lydia's scripts with modifications
            feature_patch, label_patch = self._make_patch(feature_file_ds, label_file_ds, grid_size, patch_size, x_i, y_i)

            if feature_patch is None:
                continue

            feature_patches = self._concat_patches(feature_patches, feature_patch)
            label_patches = self._concat_patches(label_patches, label_patch)

        if feature_patches is not None:
            feature_patch_path = os.path.join(feature_patches_root, str(self.run_num) + ".nc")
            label_patch_path = os.path.join(label_patches_root, str(self.run_num) + ".nc")

            feature_patches.to_netcdf(feature_patch_path)
            label_patches.to_netcdf(label_patch_path)

    
    # For getting list of chosen variables for xarray dict
    def _load_vars_list(self):
        pass

    
    def _concat_patches(self, patches, patch):
        if patches is None:
            patches = copy.deepcopy(patch.expand_dims(dim='n_samples'))
        else:
            patches = xr.concat([patches,patch.expand_dims(dim='n_samples')],dim='n_samples')

        return patches
    

    def _make_patch(self, feature_file_ds, label_file_ds, grid_size, patch_size, x, y):
        halfstep = int(patch_size/2)
        y_max = grid_size[1]-halfstep
        x_max = grid_size[0]-halfstep

        feature_patch = None
        label_patch = None

        if x >= halfstep and x < x_max and y >= halfstep and y < y_max:
            feature_patch = feature_file_ds.isel(x=slice(x-halfstep,x+halfstep),y=slice(y-halfstep,y+halfstep))
            label_patch = label_file_ds.isel(x=slice(x-halfstep,x+halfstep),y=slice(y-halfstep,y+halfstep))

        return feature_patch, label_patch
    

    # Also removes any feature file that does not have a corresponding label
    def _get_label_files(self, feature_files):
        feature_files_to_remove = []
        label_files = []
        for feature_file in feature_files:
            label_file = self._find_label_path(feature_file)
            if label_file is None:
                feature_files_to_remove.append(feature_file)
            else:
                label_files.append(label_file)
            
        for file_to_remove in feature_files_to_remove:
            feature_files.remove(file_to_remove)

        if len(feature_files) != len(label_files):
            raise Exception("Method _get_label_files has failed. Not all dud feature files removed.")

        return feature_files, label_files



    # Three possible options:
    # 1. use dates to set range
    # 2. use hard indeces to set range
    # 3. use percentages (real numbers) to set range
    # NOTE: Does not check validity of selections. May add that later
    def _select_data_range(self, file_list):
        data_start = self.config["Patches"]["data_start"]
        data_end = self.config["Patches"]["data_end"]
        use_date = self.config["Patches"]["use_date_for_data_range"]
        date_format = self.config["Input"]["date_dir_format"]
        feature_root = self.config["Input"]["examples_root"]
        feature_glob = self.config["Input"]["examples_glob"]

        glob_path = self._glob_path_maker(feature_root, feature_glob)
        feature_path_date_index = self._date_dir_index_finder(glob_path)

        date_format_ISO_8601 = "%Y-%m-%d"
        file_list = np.array(file_list)

        if use_date:
            if data_start is not None:
                start_date = datetime.datetime.strptime(data_start, date_format).strftime(date_format_ISO_8601)
                start_date = np.datetime64(start_date)
            if data_end is not None:
                end_date = datetime.datetime.strptime(data_end, date_format).strftime(date_format_ISO_8601)
                end_date = np.datetime64(end_date)

            file_date_strings = [file.split("/")[feature_path_date_index] for file in file_list]
            file_dates = [datetime.datetime.strptime(file_date_str, date_format) for file_date_str in file_date_strings]
            file_dates = np.array([file_date.strftime(date_format_ISO_8601) for file_date in file_dates], dtype="datetime64")

            if data_start is not None and data_end is not None:
                return file_list[np.where(np.logical_and(file_dates >= start_date, file_dates <= end_date))].tolist()
            elif data_start is None and data_end is not None:
                return file_list[np.where(file_dates <= end_date)].tolist()
            elif data_start is not None and data_end is None:
                return file_list[np.where(file_dates >= start_date)].tolist()
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
                

    def _find_label_path(self, feature_file):
        feature_root = self.config["Input"]["examples_root"]
        feature_glob = self.config["Input"]["examples_glob"]
        feature_hour_regex = self.config["Input"]["examples_hour_regex"]
        feature_minute_regex = self.config["Input"]["examples_minute_regex"]
        labels_root = self.config["Input"]["labels_root"]
        labels_glob = self.config["Input"]["labels_glob"]
        labels_hour_regex = self.config["Input"]["labels_hour_regex"]
        labels_minute_regex = self.config["Input"]["labels_minute_regex"]

        glob_path = self._glob_path_maker(feature_root, feature_glob)
        feature_path_date_index = self._date_dir_index_finder(glob_path)
        feature_path_array = feature_file.split("/")
        date = feature_path_array[feature_path_date_index]

        glob_path = self._glob_path_maker(labels_root, labels_glob)
        label_path_date_index = self._date_dir_index_finder(glob_path)
        glob_path_array = glob_path.split("/")
        glob_path_array[label_path_date_index] = date

        feature_filename = os.path.split(feature_file)[-1]
        feature_time = self._get_time_from_regex(feature_hour_regex, feature_minute_regex, feature_filename)

        label_glob_with_date = "/".join(glob_path_array)
        label_files = glob.glob(label_glob_with_date)

        label_path = None
        for label_file in label_files:
            label_filename = os.path.split(label_file)[-1]
            label_time = self._get_time_from_regex(labels_hour_regex, labels_minute_regex, label_filename)

            if feature_time == label_time:
                label_path = label_file
                break

        return label_path


    # def _date_dir_index_finder(self, glob_path):
    #     glob_path_array = np.array(glob_path.split("/"))
    #     date_index = np.where(glob_path_array == "*")[0][0] # WAS date_index = np.where(glob_path_array == "*")[0][0]
    #     return date_index


    # NOTE: Hour MUST be in format of %H%H and minute MUST be in format of %M%M
    def _get_time_from_regex(self, hour_reg, min_reg, filename):
        reg_hour = re.search(hour_reg, filename)
        if not reg_hour:
            raise IOError("A filename_hour_regex is not valid. Can't find hour.")
        hour = reg_hour.group(1)

        if min_reg is not None:
            reg_min = re.search(min_reg, filename)
            if not reg_min:
                raise IOError("A filename_minute_regex is not valid. Can't find minute.")
            min = reg_min.group(1)
        else:
            min = "00"
        
        time = hour + min
        return time


    # NOTE: Please make sure the start and end dates in ISO_8601 format
    def new_select_data_range(self):
        pass


    # Extracts the most possible time information from data's directory structure and file name 
    # NOTE: See the required characters for designating where each datetime component is located in each file's path
    # NOTE: If a lower level datetime unit is used (for example hours or minute), every higher level must also be present
    # NOTE: datetime_positions chars differ from the datetime.datetime chars needed for "datetime_ISO_formats". May change this later
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
        elif "h" in datetime_chars_seperated:
            dataset_date_resolution = "datetime64[h]"
        elif "D" in datetime_chars_seperated:
            dataset_date_resolution = "datetime64[D]"
        elif "M" in datetime_chars_seperated:
            dataset_date_resolution = "datetime64[M]"
        else:
            dataset_date_resolution = "datetime64[Y]"

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

        return datetimes, dataset_date_resolution


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