from matplotlib import markers
import xarray as xr
import argparse
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os
import cartopy
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
from cartopy.feature import ShapelyFeature
import cartopy.io.img_tiles as cimgt
import numpy as np
import glob
from datetime import datetime, timedelta
from custom_metrics import MaxCriticalSuccessIndex
from preprocessor import unpack_ne_dim_output
from storm_report_loader import StormReportLoader

def create_parser():
    '''
    Create argument parser
    '''
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description='Unet Preprocessing', fromfile_prefix_chars='@')

    ######## CASESTUDY SETTINGS #####################
    parser.add_argument('--init_datetime', type=str, default='2019-05-20:2000') # Was '2019-05-01:1900'
    parser.add_argument('--ens_member', type=int, default=1)
    parser.add_argument('--ens_size', type=int, default=18)
    parser.add_argument('--num_patches_per_col', type=int, default=4)
    parser.add_argument('--num_patches_per_row', type=int, default=4)
    parser.add_argument('--unprocessed_examples', type=str, default='/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour-more_fields-1_inch-cross_val/patches/animations/20190520/unprocessed/examples/0000.nc')
    ###### BOTH SETTINGS ##############
    parser.add_argument('--plot_animation', '-a', action='store_true')
    parser.add_argument('--include_reports', '-i', action='store_true')
    parser.add_argument('--plot_output_dir', type=str, default='/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour-more_fields-1_inch-cross_val/images/animations')
    parser.add_argument('--predictions_paths', type=str, default='/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour_fixed/patches/test/init_plus_00/predictions/y_hats.nc')
    # The below setting needs to be glob for graphs and direct path for animations
    parser.add_argument('--unprocessed_labels', type=str, default='/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour-more_fields-1_inch-cross_val/patches/animations/20190520/unprocessed/labels/0000.nc')
    ###### TEST DATA PLOTS SETTINGS #############
    parser.add_argument('--hailcast_files', type=str, default='/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour-128_size-more_fields/patches/animations/20190520/unprocessed/examples/0000.nc')

    return parser


def plot_casestudy_plots(args):
    predictions_paths = args["predictions_paths"]
    unprocessed_labels_path = args["unprocessed_labels"]
    unprocessed_examples_path = args["unprocessed_examples"]
    # hailcast_files = args["hailcast_files"]
    ens_member = args["ens_member"]
    ens_size = args["ens_size"]
    num_patches_per_col = args["num_patches_per_col"]
    num_patches_per_row = args["num_patches_per_row"]
    plot_output_dir = args["plot_output_dir"]
    init_datetime = args["init_datetime"]
    include_reports = args["include_reports"]

    init_datetime = datetime.strptime(init_datetime, "%Y-%m-%d:%H%M")

    ######## TEMP #################
    # predictions_paths = glob.glob("/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour-more_fields-1_inch-cross_val/patches/animations/20190501/predictions/y_hats_*")
    # predictions_paths.sort()
    # predictions_paths = [predictions_paths]
    # predictions_paths = ["/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour-more_fields-1_inch-cross_val/patches/animations/20190501/predictions/y_hats.nc"]

    # predictions_paths = ["/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour-128_size-more_fields-1_inch/patches/animations/20190520/predictions/y_hats.nc"]
    predictions_paths = ["/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour-more_fields-1_inch-cross_val/patches/animations/20190520/predictions/y_hats.nc"]
    # predictions_paths.sort()
    # predictions_paths = [predictions_paths]
    ##################################

    # NOTE: # This block had to keep decode=False for legacy reasons. So the latlons could be put directly into the numpy arrays without losing them.
    predictions_path = predictions_paths[0] # predictions_paths is always assumed to be list of either lists or strings. We expect only one element here
    if type(predictions_path) is list:
        ds_pred = xr.open_mfdataset(predictions_path, concat_dim='time_dim', combine='nested', engine='netcdf4', decode_cf=False)
    else:
        ds_pred = xr.open_dataset(predictions_path, decode_cf=False)
    # The following 3 files are assumed to be single files rather than collections of files since this is for one case study
    ds_labels = xr.open_dataset(unprocessed_labels_path, decode_cf=False)
    ds_examples = xr.open_dataset(unprocessed_examples_path, decode_cf=False)
    ds_hailcast = xr.open_dataset(unprocessed_examples_path, decode_cf=False)

    ds_examples = ds_examples[{"ne": ens_member}]
    ds_examples = ds_examples.drop("time")
    ds_hailcast = ds_hailcast[{"ne": ens_member}]
    ds_hailcast = ds_hailcast.drop("time")

    ds_pred = ds_pred.transpose("n_samples", "lat_dim", "lon_dim", "time_dim")
    ds_labels = ds_labels.transpose("n_samples", "lat_dim", "lon_dim", "time_dim")
    ds_examples = ds_examples.transpose("n_samples", "lat_dim", "lon_dim", "time_dim")
    ds_hailcast = ds_hailcast.transpose("n_samples", "lat_dim", "lon_dim", "time_dim")

    # Because we are doing array splitting we need to convert everything to numpy arrays and therefore need to keep track of our keys and dims
    var_keys_pred = {key_str:i for i,key_str in enumerate(ds_pred.keys())}
    var_keys_labels = {key_str:i for i,key_str in enumerate(ds_labels.keys())}
    var_keys_examples = {key_str:i for i,key_str in enumerate(ds_examples.keys())}
    var_keys_hailcast = {key_str:i for i,key_str in enumerate(ds_hailcast.keys())}

    # corner_patch_index = ens_member * num_patches_per_col * num_patches_per_row

    for time_index in range(12): # number of time steps
        selected_patches = np.arange(ens_member, ens_size * num_patches_per_col * num_patches_per_row + ens_member, ens_size)

        ds_pred_np = ds_pred[{"time_dim": time_index}].to_array().to_numpy()
        ds_labels_np = ds_labels[{"time_dim": time_index}].to_array().to_numpy()
        ds_examples_np = ds_examples[{"time_dim": time_index}].to_array().to_numpy()
        ds_hailcast_np = ds_hailcast[{"time_dim": time_index}].to_array().to_numpy()

        ds_pred_np_one_img = np.take(ds_pred_np, selected_patches, axis=1)
        ds_hailcast_np_one_img = np.take(ds_hailcast_np, np.arange(num_patches_per_col * num_patches_per_row), axis=1)
        ds_labels_np_one_img = np.take(ds_labels_np, np.arange(num_patches_per_col * num_patches_per_row), axis=1)
        ds_examples_np_one_img = np.take(ds_examples_np, np.arange(num_patches_per_col * num_patches_per_row), axis=1)

        # Stitch together all patches from inside the netcdf file.
        # Assumes patches are in order
        split_pred_ds_list = np.split(ds_pred_np_one_img, indices_or_sections=num_patches_per_col, axis=1)
        split_hailcast_ds_list = np.split(ds_hailcast_np_one_img, indices_or_sections=num_patches_per_col, axis=1)
        split_labels_ds_list = np.split(ds_labels_np_one_img, indices_or_sections=num_patches_per_col, axis=1)
        split_examples_ds_list = np.split(ds_examples_np_one_img, indices_or_sections=num_patches_per_col, axis=1)
        split_pred_ds_list_concat = []
        split_hailcast_ds_list_concat = []
        split_labels_ds_list_concat = []
        split_examples_ds_list_concat = []
        for split_pred_ds, split_hailcast_ds, split_labels_ds, split_examples_ds in zip(split_pred_ds_list, split_hailcast_ds_list, split_labels_ds_list, split_examples_ds_list):
            split_pred_ds_list_concat.append(np.concatenate(np.split(split_pred_ds, indices_or_sections=num_patches_per_row, axis=1), axis=2))
            split_hailcast_ds_list_concat.append(np.concatenate(np.split(split_hailcast_ds, indices_or_sections=num_patches_per_row, axis=1), axis=2))
            split_labels_ds_list_concat.append(np.concatenate(np.split(split_labels_ds, indices_or_sections=num_patches_per_row, axis=1), axis=2))
            split_examples_ds_list_concat.append(np.concatenate(np.split(split_examples_ds, indices_or_sections=num_patches_per_row, axis=1), axis=2))
        complete_pred_ds = np.concatenate(split_pred_ds_list_concat, axis=3)
        complete_pred_ds = np.take(complete_pred_ds, 0, axis=1)
        complete_hailcast_ds = np.concatenate(split_hailcast_ds_list_concat, axis=3)
        complete_hailcast_ds = np.take(complete_hailcast_ds, 0, axis=1)
        complete_labels_ds = np.concatenate(split_labels_ds_list_concat, axis=3)
        complete_labels_ds = np.take(complete_labels_ds, 0, axis=1)
        complete_examples_ds = np.concatenate(split_examples_ds_list_concat, axis=3)
        complete_examples_ds = np.take(complete_examples_ds, 0, axis=1)

        # Select our variables for plotting including latlons
        lons = np.take(complete_examples_ds, var_keys_examples["lon"], axis=0)
        lats = np.take(complete_examples_ds, var_keys_examples["lat"], axis=0)
        comp_dz = np.take(complete_examples_ds, var_keys_examples["comp_dz"], axis=0)
        pred_hail = np.take(complete_pred_ds, var_keys_pred["MESH_class_bin"], axis=0)
        true_hail = np.take(complete_labels_ds, var_keys_labels["MESH_class_bin"], axis=0)
        hailcast = np.take(complete_hailcast_ds, var_keys_hailcast["hailcast"], axis=0)

        print('Timestep ' + str(time_index) + " max: " + str(np.max(pred_hail)))

        domain_refl_plot(plot_output_dir, lons, lats, comp_dz, pred_hail, init_datetime, include_reports, time_index)
        domain_truth_plot(plot_output_dir, lons, lats, true_hail, pred_hail, hailcast, init_datetime, include_reports, time_index)


def domain_refl_plot(plot_output_dir, lons, lats, comp_dz, pred_val, init_datetime, include_reports, plot_num = 0):
    #Create a custom colormap for dBZ
    blues = np.repeat(1,51)
    reds = np.linspace(0.25,0, num = 51)
    greens = np.linspace(1,0, num = 51)
    my_cmap = np.column_stack([reds,greens])
    my_cmap = np.column_stack([my_cmap,blues])
    blues = np.repeat(0,51)
    reds = np.repeat(0,51)
    greens = np.linspace(1,0.65, num = 51)
    new_color = np.column_stack([reds,greens])
    new_color = np.column_stack([new_color,blues])
    my_cmap = np.append(my_cmap,new_color, axis = 0)
    blues = np.repeat(0,71)
    reds = np.repeat(1,71)
    greens = np.linspace(1,0.52, num = 71)
    new_color = np.column_stack([reds,greens])
    new_color = np.column_stack([new_color,blues])
    my_cmap = np.append(my_cmap,new_color, axis = 0)
    blues = np.repeat(0,51)
    reds = np.linspace(1,0.6, num = 51)
    greens = np.repeat(0,51)
    new_color = np.column_stack([reds,greens])
    new_color = np.column_stack([new_color,blues])
    my_cmap = np.append(my_cmap,new_color, axis = 0)
    blues = np.linspace(1,0.5, num = 31)
    reds = np.linspace(1,0.5, num = 31)
    greens = np.repeat(0,31)
    new_color = np.column_stack([reds,greens])
    new_color = np.column_stack([new_color,blues])
    my_cmap = np.append(my_cmap,new_color, axis = 0)
    my_cmap = colors.ListedColormap(my_cmap, name="refl")
    ##################################

    # Titles and labels
    current_datetime = init_datetime + timedelta(minutes=plot_num*5) # *5 for 5 minutes
    init_time_offset_str = "{:02d}".format(plot_num*5)
    title = "Severe Hail Event on " + current_datetime.strftime("%Y-%m-%d:%H%M") + " UTC (Initialization Time + " + init_time_offset_str + " Minutes)"
    comp_dz_label = "WoFS Composite Reflectivity (dBZ > 5)"
    pred_val_label = "Predicted Probability of Hail > 10 mm"

    # Setup figure
    fig = plt.figure(figsize=(16, 12), dpi=300)
    fig.patch.set_facecolor('white')
    plt.tight_layout()
    ax = plt.axes(projection=ccrs.PlateCarree())
    plt.title(title, fontsize=20)

    ############ May 1 bounds ####################
    # Full domain
    # ax.set_xlim((-101, -95.5))
    # ax.set_ylim((32, 36))
    # zoomed
    # ax.set_xlim((-101.2, -97.2))
    # ax.set_ylim((32.2, 35))
    # more zoomed
    # ax.set_xlim((-100.2, -98.2))
    # ax.set_ylim((32.8, 34))
    ############ May 20 bounds ######################
    # Full domain
    # ax.set_xlim((-102.5, -96.8))
    # ax.set_ylim((32, 37))
    # zoomed
    ax.set_xlim((-101.2, -98.2))
    ax.set_ylim((33.2, 36))
    ########################################

    # The following block sets up all background features
    ax.add_feature(cartopy.feature.NaturalEarthFeature(
        'cultural', 'admin_1_states_provinces_lines', '50m',
        edgecolor='gray', facecolor='none'))
    ax.add_feature(cartopy.feature.NaturalEarthFeature(
        'cultural', 'admin_0_countries', '50m',
        edgecolor='gray', facecolor='none'))
    ax.add_feature(cartopy.feature.NaturalEarthFeature(
        'cultural', 'admin_2_counties', '10m',
        edgecolor='gray', facecolor='none', alpha=0.2))
    # Make interstate feature
    reader = shpreader.Reader("/home/tgschmidt/shapefiles/tl_2016_us_primaryroads/tl_2016_us_primaryroads.shp")
    names = []
    geoms = []
    for rec in reader.records():
        if (rec.attributes['FULLNAME'][0]=='I'):
            names.append(rec)
            geoms.append(rec.geometry)
    shape_feature = ShapelyFeature(geoms, ccrs.PlateCarree(), edgecolor='darkblue', lw=0.4,facecolor='none')
    ax.add_feature(shape_feature)
    # Add stamen background in grayscale
    stamen_terrain = cimgt.Stamen('terrain-background', desired_tile_form='L')
    ax.add_image(stamen_terrain,11, cmap='gray')

    # Add actual plots
    plt.contourf(lons,lats,comp_dz,np.arange(5, 76, 1), vmin = 5, vmax = 75, cmap=my_cmap, transform=ccrs.PlateCarree(), zorder=10)
    plt.colorbar(fraction=0.043, pad=0.07, ticks = np.arange(5,76,5)).set_label(label=comp_dz_label,size=30)
    # was np.arange(0,101,10)/100 and vmin 0/1
    plt.contour(lons,lats,pred_val,np.arange(0,21,2)/100, colors='k', transform=ccrs.PlateCarree(), linewidths=2.3, vmin=0, vmax=0.2, zorder=11)
    plt.colorbar(fraction=0.043, pad=0.02, ticks = np.arange(0,21,2)/100).set_label(label=pred_val_label,size=30)

    if include_reports:
        loader = StormReportLoader("/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/storm_reports/20190501/STORM_EVENTS_2019-2019.csv", "NOAA", init_datetime.strftime("%Y%m%d%H%M"), plot_num*5, 0)

        storm_reports_hail_lats, storm_reports_hail_lons = loader.get_hail_reports()
        storm_reports_hail_sig_lats, storm_reports_hail_sig_lons = loader.get_hail_reports('sig_severe')

        plt.scatter(storm_reports_hail_lons, storm_reports_hail_lats, color='k', s=180, marker='o', edgecolors= "black", transform=ccrs.PlateCarree(), zorder=12, label="Severe Hail")
        plt.scatter(storm_reports_hail_sig_lons, storm_reports_hail_sig_lats, color='k', s=700, marker='*', edgecolors= "black", transform=ccrs.PlateCarree(), zorder=13, label="Sig Severe Hail")
        plt.legend(prop={'size': 25}, loc=4).set_zorder(99)

    plt.savefig(os.path.join(plot_output_dir, "refl_plot_" + "{:04d}".format(plot_num) + '.png'), bbox_inches='tight')


def domain_truth_plot(plot_output_dir, lons, lats, true_val, pred_val, hailcast, init_datetime, include_reports, plot_num = 0):
    # Titles and labels
    current_datetime = init_datetime + timedelta(minutes=plot_num*5) # *5 for 5 minutes
    init_time_offset_str = "{:02d}".format(plot_num*5)
    title = "Severe Hail Event on " + current_datetime.strftime("%Y-%m-%d:%H%M") + " UTC (Initialization Time + " + init_time_offset_str + " Minutes)"
    real_val_label = "GridRad Location of Hail > 10 mm"
    pred_val_label = "Deterministic Predicted Probability of Hail > 10 mm"
    hailcast_label = "Deterministic WoFS Hailcast Location of Hail > 10 mm"

    hailcast_bin = np.zeros(hailcast.shape)
    hailcast_bin[hailcast > 1] = 1 # was 0.393701
    hailcast = hailcast_bin
    hailcast[np.nonzero(hailcast == 0)] = np.nan

    # Setup figure
    fig = plt.figure(figsize=(16, 12), dpi=300)
    fig.patch.set_facecolor('white')
    plt.tight_layout()
    ax = plt.axes(projection=ccrs.PlateCarree())
    plt.title(title, fontsize=20)

    ######## May 1 bounds ####################
    # Full domain
    # ax.set_xlim((-101, -95.5))
    # ax.set_ylim((32, 36))
    # zoomed
    # ax.set_xlim((-101.2, -97.2))
    # ax.set_ylim((32.2, 35))
    # more zoomed
    # ax.set_xlim((-100.2, -98.2))
    # ax.set_ylim((32.8, 34))
    ############ May 20 bounds ######################
    # Full domain
    # ax.set_xlim((-102.5, -96.8))
    # ax.set_ylim((32, 37))
    # zoomed
    ax.set_xlim((-101.2, -98.2))
    ax.set_ylim((33.2, 36))
    ########################################

    # The following block sets up all background features
    ax.add_feature(cartopy.feature.NaturalEarthFeature(
        'cultural', 'admin_1_states_provinces_lines', '50m',
        edgecolor='gray', facecolor='none'))
    ax.add_feature(cartopy.feature.NaturalEarthFeature(
        'cultural', 'admin_0_countries', '50m',
        edgecolor='gray', facecolor='none'))
    ax.add_feature(cartopy.feature.NaturalEarthFeature(
        'cultural', 'admin_2_counties', '10m',
        edgecolor='gray', facecolor='none', alpha=0.2))
    # Make interstate feature
    reader = shpreader.Reader("/home/tgschmidt/shapefiles/tl_2016_us_primaryroads/tl_2016_us_primaryroads.shp")
    names = []
    geoms = []
    for rec in reader.records():
        if (rec.attributes['FULLNAME'][0]=='I'):
            names.append(rec)
            geoms.append(rec.geometry)
    shape_feature = ShapelyFeature(geoms, ccrs.PlateCarree(), edgecolor='darkblue', lw=0.4,facecolor='none')
    ax.add_feature(shape_feature)
    # Add stamen background in grayscale
    stamen_terrain = cimgt.Stamen('terrain-background', desired_tile_form='L')
    ax.add_image(stamen_terrain,11, cmap='gray')

    # Add actual plots
    true_val[np.nonzero(true_val == 0)] = np.nan
    plt.contourf(lons,lats,true_val, [0, 1], colors='red', transform=ccrs.PlateCarree(), zorder=10, alpha=0.8)
    plt.contourf(lons,lats,hailcast, [0, 1], colors='blue', transform=ccrs.PlateCarree(), zorder=11, alpha=0.8)
    # was np.arange(10,101,10)/100 and vmin/vmax 0.1/1
    plt.contourf(lons,lats,pred_val,np.arange(2,21,2)/100, cmap="Greens", transform=ccrs.PlateCarree(), vmin=0.02, vmax=0.2, zorder=12, alpha=0.8)
    plt.colorbar(fraction=0.043, pad=0.02, ticks = np.arange(2,21,2)/100).set_label(label=pred_val_label,size=30) #Changed padding here since other colorbars removed


    if include_reports:
        loader = StormReportLoader("/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/storm_reports/20190501/STORM_EVENTS_2019-2019.csv", "NOAA", init_datetime.strftime("%Y%m%d%H%M"), plot_num*5, 0)

        storm_reports_hail_lats, storm_reports_hail_lons = loader.get_hail_reports()
        storm_reports_hail_sig_lats, storm_reports_hail_sig_lons = loader.get_hail_reports('sig_severe')

        plt.scatter(storm_reports_hail_lons, storm_reports_hail_lats, color='k', s=180, marker='o', edgecolors= "black", transform=ccrs.PlateCarree(), zorder=13, label="Severe Hail")
        plt.scatter(storm_reports_hail_sig_lons, storm_reports_hail_sig_lats, color='k', s=700, marker='*', edgecolors= "black", transform=ccrs.PlateCarree(), zorder=14, label="Sig Severe Hail")
        plt.legend(prop={'size': 25}, loc=4).set_zorder(99)

    plt.savefig(os.path.join(plot_output_dir, "truth_plot_" + "{:04d}".format(plot_num) + '.png'), bbox_inches='tight')


# NOTE: This is only for binary case
def plot_test_data_plots(args):
    hailcast_files = args["hailcast_files"]
    labels_glob = args["unprocessed_labels"]
    plot_output_dir = args["plot_output_dir"]
    # predictions_paths = args["predictions_paths"]
    predictions_paths = [
                        # ["/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour-128_size-more_fields/patches/test/predictions/y_hats_single_00.nc",
                        #  "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour-128_size-more_fields/patches/test/predictions/y_hats_single_15.nc",
                        #  "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour-128_size-more_fields/patches/test/predictions/y_hats_single_30.nc",
                        #  "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour-128_size-more_fields/patches/test/predictions/y_hats_single_45.nc",
                        #  "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour-128_size-more_fields/patches/test/predictions/y_hats_single_55.nc"],
                        # ["/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour-128_size-more_fields/patches/test/predictions/y_hats_00.nc",
                        #  "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour-128_size-more_fields/patches/test/predictions/y_hats_15.nc",
                        #  "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour-128_size-more_fields/patches/test/predictions/y_hats_30.nc",
                        #  "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour-128_size-more_fields/patches/test/predictions/y_hats_45.nc",
                        #  "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour-128_size-more_fields/patches/test/predictions/y_hats_55.nc"],
                        "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-1_hour-1_inch-train_val_test-cross_val/patches/test/predictions/y_hats.nc"]
                        # "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour-more_fields-1_inch-cross_val/patches/test/predictions/y_hats_flattened.nc"]
                        # "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour-more_fields-1_inch-cross_val/patches/cv_folds/fold_0000/log_reg_files/predictions/y_hats.nc"]
                        # "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour-more_fields-1_inch-balanced/patches/test_split/predictions/y_hats_ens_2.nc"]
    # model_names = ["Deterministic Single 2D UNet", "Deterministic Multiple 2D UNets", "Deterministic Time-Resolving 3D UNet"]
    model_names = ["Deterministic Time-Resolving 3D UNet"] #, "UH 2-5 Logistic Regression"]#, "Ensemble Time-Resolving 3D UNet"]
    hail_name = "MESH_class_bin"
    lead_time_names_pretty = ["Init Plus 00", "Init Plus 15", "Init Plus 30", "Init Plus 45", "Init Plus 60"]
    lead_time_minutes = [0, 15, 30, 45, 55]
    lead_time_indices = [0, 3, 6, 9, 11]

    # Load all the xarray dataset objects
    hailcast_list = glob.glob(hailcast_files)
    hailcast_list.sort()
    hailcast_dataset = xr.open_mfdataset(hailcast_list, concat_dim='n_samples', combine='nested', engine='netcdf4')
    labels_list = glob.glob(labels_glob)
    labels_list.sort()
    labels_dataset = xr.open_mfdataset(labels_list, concat_dim='n_samples', combine='nested', engine='netcdf4')

    labels_dataset = labels_dataset.transpose("n_samples", "lat_dim", "lon_dim", "time_dim")

    # Load predictions and truths for each of the valid times
    truths_flattened = []
    predictions_flattened = []
    for j, predictions_path in enumerate(predictions_paths):
        predictions = []
        truths = []

        for i, lead_time_index in enumerate(lead_time_indices):
            if type(predictions_path) is list:
                predictions.append(xr.open_dataset(predictions_path[i])[hail_name].to_numpy().ravel())
            else:
                predictions.append(xr.open_dataset(predictions_path)[{"time_dim": lead_time_index}][hail_name].to_numpy().ravel())
            
            if "Ensemble" not in model_names[j]:
                truth = unpack_ne_dim_output(labels_dataset[{"time_dim": lead_time_index}][hail_name].to_numpy(), 18, 1).ravel()
            else:
                truth = labels_dataset[{"time_dim": lead_time_index}][hail_name].to_numpy().ravel()
            truths.append(truth)
        
        truths_flattened.append(truths)
        predictions_flattened.append(predictions)

    # Load a hailcast list
    hailcast_flattened = []
    for lead_time_index in lead_time_indices:
        hailcast_flattened.append(hailcast_dataset[{"time_dim": lead_time_index}]['hailcast'].to_numpy().ravel())

    # Calculate max csi, calibration, and ROC for model data
    roc_auc_scores = []
    roc_curves = []
    max_csis = []
    calibration_curves = []
    for truths, predictions in zip(truths_flattened, predictions_flattened):
        roc_auc_score_one_model = []
        roc_curve_one_model = []
        max_csi_one_model = []
        calibration_curve_one_model = []
        
        for truth, prediction in zip(truths, predictions):
            roc_auc_score_one_model.append(roc_auc_score(truth, prediction))

            fpr, tpr, thresholds = roc_curve(truth, prediction)
            roc_curve_one_model.append((fpr, tpr))

            max_csi_metric = MaxCriticalSuccessIndex()
            max_csi_metric.reset_state()
            max_csi_metric.update_state(truth, prediction)
            max_csi_one_model.append(max_csi_metric.result().numpy())

            prob_true, prob_pred = calibration_curve(truth, prediction, n_bins=100)
            calibration_curve_one_model.append((prob_pred, prob_true))

        roc_auc_scores.append(roc_auc_score_one_model)
        roc_curves.append(roc_curve_one_model)
        max_csis.append(max_csi_one_model)
        calibration_curves.append(calibration_curve_one_model)

    # Calculate max csi, calibration, and ROC for hailcast
    roc_auc_scores_hailcast = []
    roc_curves_hailcast = []
    max_csis_hailcast = []
    calibration_curves_hailcast = []
    # TODO: Make the below into something that intelligently finds which truths_flattened to use (hailcast currently is always "unpacked")
    # rather than taking harcoded truths_flattened[0] because it is the one I happened to set that isn't an ensemble
    for truth, hailcast in zip(truths_flattened[0], hailcast_flattened):
        roc_auc_scores_hailcast.append(roc_auc_score(truth, hailcast))

        fpr, tpr, thresholds = roc_curve(truth, hailcast)
        roc_curves_hailcast.append((fpr, tpr))

        max_csi_metric = MaxCriticalSuccessIndex()
        max_csi_metric.reset_state()
        max_csi_metric.update_state(truth, hailcast)
        max_csis_hailcast.append(max_csi_metric.result().numpy())

        prob_true, prob_pred = calibration_curve(truth, hailcast, n_bins=100)
        calibration_curves_hailcast.append((prob_pred, prob_true))

    # Make the max CSI plot
    fig = plt.figure(figsize=(16, 12), dpi=300)
    fig.patch.set_facecolor('white')
    for i, max_csi in enumerate(max_csis):
        plt.plot(lead_time_minutes, max_csi, linestyle='-', label=model_names[i], marker='o', linewidth=6, markersize=20)
    plt.plot(lead_time_minutes, max_csis_hailcast, linestyle='--', label="Ensemble Probabilistic Hailcast", marker='o', linewidth=6, markersize=20)
    plt.ylabel('Pixelwise Max CSI', size=48)
    plt.yticks(fontsize=28)
    plt.xlabel('Time Since WoFS Init (mins)', size=48)
    plt.xticks(fontsize=28)
    plt.title('Test Set Pixelwise Max CSI With Forecast Runtime', size=48)
    plt.legend(prop={'size': 25})
    plt.tight_layout()
    plt.savefig(os.path.join(plot_output_dir, 'max_csi.png'), bbox_inches='tight')

    # Make the ROC plot
    fig = plt.figure(figsize=(16, 12), dpi=300)
    fig.patch.set_facecolor('white')
    plt.plot([0,1], [0,1], linestyle='--', c='k', label='No Skill: AUC=%.3f' % (0.5))
    # TODO: REMOVE HARDCODED "0's" in if
    for roc_auc_score_value, roc_auc_score_hailcast_value, roc_curve_value, roc_curve_hailcast_value, lead_time_name in zip(roc_auc_scores[0], roc_auc_scores_hailcast, roc_curves[0], roc_curves_hailcast, lead_time_names_pretty):
        plt.plot(roc_curve_value[0], roc_curve_value[1], linestyle='-', linewidth=6, label="ML Hail " + lead_time_name + ': AUC=%.3f' % (roc_auc_score_value))
        plt.plot(roc_curve_hailcast_value[0], roc_curve_hailcast_value[1], linestyle='-.', linewidth=6, label="Hailcast " + lead_time_name + ': AUC=%.3f' % (roc_auc_score_hailcast_value), color=plt.gca().lines[-1].get_color())
    plt.xlabel('False Positive Rate', size=48)
    plt.ylabel('True Positive Rate', size=48)
    plt.title('Receiver Operating Characteristic Curve', size=48)
    plt.legend(prop={'size': 25})
    plt.tight_layout()
    plt.savefig(os.path.join(plot_output_dir, 'ROC_Curve.png'), bbox_inches='tight')

    # Make the calibration plot
    fig = plt.figure(figsize=(16, 12), dpi=300)
    fig.patch.set_facecolor('white')
    plt.plot([0,1], linestyle='--', c='k')
    # TODO: REMOVE HARDCODED "0's" in if
    for calibration_curve_val, calibration_curve_hailcast_val, lead_time_name in zip(calibration_curves[0], calibration_curves_hailcast, lead_time_names_pretty):
        plt.plot(calibration_curve_val[0], calibration_curve_val[1], linestyle='-', linewidth=6, label="ML Hail " + lead_time_name)
        plt.plot(calibration_curve_hailcast_val[0], calibration_curve_hailcast_val[1], linestyle='-.', linewidth=6, label="Hailcast " + lead_time_name, color=plt.gca().lines[-1].get_color())
    plt.ylabel("Observed Frequency", size=48)
    plt.xlabel("Predicted Probability", size=48)
    plt.title("Reliability Diagram", size=48)
    plt.legend(loc="best", prop={'size': 25})
    plt.tight_layout()
    plt.savefig(os.path.join(plot_output_dir, 'Reliability_Diagram.png'), bbox_inches='tight')


if __name__ == "__main__":
    # Perform all the argument parsing
    parser = create_parser()
    args = parser.parse_args()
    args = vars(args)

    plot_animation = args["plot_animation"]

    if plot_animation:
        plot_casestudy_plots(args)
    else:
        plot_test_data_plots(args)