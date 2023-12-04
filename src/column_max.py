import xarray as xr
import argparse
import glob
import os
import numpy as np


def create_parser():
    '''
    Create argument parser
    '''
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description='', fromfile_prefix_chars='@')

    parser.add_argument('--input_year_dir_glob', type=str, default="/ourdisk/hpc/ai2es/gridrad_severe/volumes/2020/*")
    parser.add_argument('--output_dir', type=str, default="/ourdisk/hpc/ai2es/severe_nowcasting/gridrad_gridded/2020")
    parser.add_argument('--n_parallel_runs', type=int, default=None)
    parser.add_argument('--run_num', type=int, default=0)

    return parser


parser = create_parser()
args = parser.parse_args()
args = vars(args)

input_year_dir_glob = args["input_year_dir_glob"]
output_dir = args["output_dir"]
n_parallel_runs = args["n_parallel_runs"]
run_num = args["run_num"]

date_dir_list = glob.glob(input_year_dir_glob)
date_dir_list.sort()

if n_parallel_runs is not None:
    date_dir_list = np.array_split(np.array(date_dir_list, dtype=object), n_parallel_runs)[run_num]

for date_path in date_dir_list:
    input_file_glob = os.path.join(date_path, "*")
    input_file_list = glob.glob(input_file_glob)
    input_file_list.sort()

    for file_path in input_file_list:
        ds = xr.open_dataset(file_path)

        file_name = file_path.split("/")[-1]
        date = date_path.split("/")[-1]
        output_path = os.path.join(output_dir, date)
        os.makedirs(output_path, exist_ok = True) 
        output_path = os.path.join(output_path, file_name)

        ds = ds.fillna(0)
        ds = ds.max(dim = "Altitude", skipna=True)
        ds = ds[{"time": 0}]
        ds = ds[["ZH"]]
        ds.to_netcdf(output_path)
        ds.close()