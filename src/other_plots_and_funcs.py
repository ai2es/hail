import matplotlib.pyplot as plt
import glob
import xarray as xr
import numpy as np

###################
ALL_LABELS_GLOB = "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour-more_fields-1_inch/patches/train/labels/*"
OUTPUT_PATH = "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour-more_fields-1_inch-cross_val/images/test_dataset_plots/hist_comp_dz_test.png"
DATA_HAD_NE_DIM = False
###################
TRAIN_LABELS_PATH_GLOB = "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour-more_fields-1_inch-balanced/patches/train/labels/*"
###################
PATH_GLOB = "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/single_model/predictions_ensemble/*"
VAR_NAME = "comp_dz" # was "srh_0to3"
###################
OUTFILE = "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/single_model/single_preds/y_hats_ens.nc"
######################


def make_hist_plots_labels():
    all_label_files = glob.glob(ALL_LABELS_GLOB)
    all_label_files.sort()

    ds = xr.open_mfdataset(all_label_files, concat_dim='n_samples', combine='nested', engine='netcdf4')
    gridrad_mesh = ds["MESH95"].to_numpy().flatten()
    gridrad_bin = ds["MESH_class_bin"].to_numpy().flatten()

    if DATA_HAD_NE_DIM:
        gridrad_mesh = gridrad_mesh[..., np.newaxis]
        gridrad_mesh = np.repeat(gridrad_mesh, 18, axis=-1)
        gridrad_mesh = gridrad_mesh.flatten()
        gridrad_bin = gridrad_bin[..., np.newaxis]
        gridrad_bin = np.repeat(gridrad_bin, 18, axis=-1)
        gridrad_bin = gridrad_bin.flatten()

    fig = plt.figure(figsize=(16, 12), dpi=300)
    fig.patch.set_facecolor('white')
    plt.hist(gridrad_mesh, [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100])
    plt.ylim((0,1000000))
    plt.axvline(25.4, color='k', linewidth=3) # 1 inch threshold
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, bbox_inches='tight')


def make_hist_plots_examples():
    all_files = glob.glob(PATH_GLOB)
    all_files.sort()

    ds = xr.open_mfdataset(all_files, concat_dim='n_samples', combine='nested', engine='netcdf4')
    data = ds[VAR_NAME].to_numpy().flatten()

    fig = plt.figure(figsize=(16, 12), dpi=300)
    fig.patch.set_facecolor('white')
    # Did have np.arange(-2000,5000,100) for srh
    plt.hist(data, [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90], log=True, edgecolor = "black")
    plt.ylim((0,10**10)) # Was 10**9 for srh
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, bbox_inches='tight')


def find_balanced_base_rate(path_glob):
    file_list = glob.glob(path_glob)
    file_list.sort()
    label_ds = xr.open_mfdataset(file_list, concat_dim='n_samples', combine='nested', engine='netcdf4')

    MESH_class = label_ds["MESH_class_bin"].to_numpy()

    counts = np.sum(MESH_class, axis=tuple(np.arange(1,len(MESH_class.shape))))
    num_30 = len(counts[np.nonzero(counts>=30)])
    print(np.sum(counts)/num_30)


def find_total_base_rate(path_glob):
    file_list = glob.glob(path_glob)
    file_list.sort()
    label_ds = xr.open_mfdataset(file_list, concat_dim='n_samples', combine='nested', engine='netcdf4')

    gridrad_bin = label_ds["MESH_class_bin"].to_numpy().flatten()

    if DATA_HAD_NE_DIM:
        gridrad_bin = gridrad_bin[..., np.newaxis]
        gridrad_bin = np.repeat(gridrad_bin, 18, axis=-1)
        gridrad_bin = gridrad_bin.flatten()
    
    print(np.sum(gridrad_bin)/len(gridrad_bin))


def check_num_of_storm_days(path_glob):
    file_list = glob.glob(path_glob)
    file_list.sort()
    label_ds = xr.open_mfdataset(file_list, concat_dim='n_samples', combine='nested', engine='netcdf4')

    all_times = np.sort(label_ds["time"].to_numpy().flatten())

    previous_time = np.datetime64('2000-01-01')
    num_of_storm_days = 0
    for time in all_times:
        if time - np.timedelta64(6,'h') >= previous_time:
            num_of_storm_days = num_of_storm_days + 1
        
        previous_time = time
    
    print(num_of_storm_days)


def combine_netcdfs_together(path_glob, outfile):
    file_list = glob.glob(path_glob)
    file_list.sort()
    ds = xr.open_mfdataset(file_list, concat_dim='time_dim', combine='nested', engine='netcdf4')

    ds.to_netcdf(outfile)


combine_netcdfs_together(PATH_GLOB, OUTFILE)