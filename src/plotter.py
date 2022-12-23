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

def create_parser():
    '''
    Create argument parser
    '''
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description='Unet Preprocessing', fromfile_prefix_chars='@')

    parser.add_argument('--predictions_path', type=str, default='/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour_fixed/patches/test/init_plus_00/predictions/y_hats.nc')
    parser.add_argument('--processed_examples', type=str, default='/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour_fixed-128_size/patches/test/processed/examples/*')
    parser.add_argument('--unprocessed_labels', type=str, default='/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour_fixed-128_size/patches/test/unprocessed/labels/*')
    # parser.add_argument('--test_data_dir', type=str, default='/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/trained_at_init_time/patches/test')
    # parser.add_argument('--truths_path', type=str, default='/Users/tschmidt/repos/hail/data/predictions/no_refl_trained_at_init_time_2022_08_03/y_hats.nc')
    # parser.add_argument('--other_features_path', type=str, default='/Users/tschmidt/repos/hail/data/predictions/no_refl_trained_at_init_time_2022_08_03/y_hats.nc')
    parser.add_argument('--plot_output_dir', type=str, default='/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour_fixed-128_size/images/test_dataset_plots')
    parser.add_argument('--init_datetime', type=str, default='2019-05-01:1900')
    # parser.add_argument('--class_names', type=str, nargs='+', default=['Hail'])
    parser.add_argument('--ens_member', type=int, default=1)
    parser.add_argument('--ens_size', type=int, default=18)
    parser.add_argument('--num_patches_per_col', type=int, default=4)
    parser.add_argument('--num_patches_per_row', type=int, default=4)
    parser.add_argument('--plot_animation', '-a', action='store_true')

    return parser


def plot_casestudy_plots(args):
    predictions_path = args["predictions_path"]
    unprocessed_labels_path = args["unprocessed_labels"]
    unprocessed_examples_path = args["unprocessed_examples"]
    ens_member = args["ens_member"]
    ens_size = args["ens_size"]
    num_patches_per_col = args["num_patches_per_col"]
    num_patches_per_row = args["num_patches_per_row"]
    plot_output_dir = args["plot_output_dir"]
    init_datetime = args["init_datetime"]

    unprocessed_labels = glob.glob(unprocessed_labels_path)
    unprocessed_labels.sort()
    unprocessed_examples = glob.glob(unprocessed_examples_path)
    unprocessed_examples.sort()

    init_datetime = datetime.strptime(init_datetime, "%Y-%m-%d:%H%M")

    ds_pred = xr.open_dataset(predictions_path, decode_cf=False) #TODO: REMOVE decode_cf when possible
    ds_labels = xr.open_mfdataset(unprocessed_labels, concat_dim='n_samples', combine='nested', engine='netcdf4', decode_cf=False) #TODO: REMOVE decode_cf when possible
    ds_examples = xr.open_mfdataset(unprocessed_examples, concat_dim='n_samples', combine='nested', engine='netcdf4', decode_cf=False) #TODO: REMOVE decode_cf when possible

    # Because we are doing array splitting we need to convert everything to numpy arrays and therefore need to keep track of our keys and dims
    var_keys_pred = {key_str:i for i,key_str in enumerate(ds_pred.keys())}
    dim_keys_pred = {key_str:i+1 for i,key_str in enumerate(list(ds_pred.dims))}
    var_keys_labels = {key_str:i for i,key_str in enumerate(ds_labels.keys())}
    dim_keys_labels = {key_str:i+1 for i,key_str in enumerate(list(ds_labels.dims))}
    var_keys_examples = {key_str:i for i,key_str in enumerate(ds_examples.keys())}
    dim_keys_examples = {key_str:i+1 for i,key_str in enumerate(list(ds_examples.dims))}

    ds_pred_np = ds_pred.to_array().to_numpy()
    ds_labels_np = ds_labels.to_array().to_numpy()
    ds_examples_np = ds_examples.to_array().to_numpy()

    # selected_patches = np.zeros(num_patches_per_col*num_patches_per_row, dtype=np.int64)
    last_patch_index = ens_member * num_patches_per_col * num_patches_per_row
    plot_num = 0

    while last_patch_index < ds_pred_np.shape[dim_keys_pred["n_samples"]]:
        # for i in range(len(selected_patches)):
        #     selected_patches[i] = last_patch_index
        selected_patches = np.arange(last_patch_index, last_patch_index + num_patches_per_col * num_patches_per_row)
        last_patch_index = last_patch_index + ens_size * num_patches_per_col * num_patches_per_row

        ds_pred_np_one_img = np.take(ds_pred_np, selected_patches, axis=dim_keys_pred["n_samples"])
        ds_labels_np_one_img = np.take(ds_labels_np, selected_patches, axis=dim_keys_labels["n_samples"])
        ds_examples_np_one_img = np.take(ds_examples_np, selected_patches, axis=dim_keys_examples["n_samples"])

        # Stitch together all patches from inside the netcdf file.
        # Assumes patches are in order
        split_pred_ds_list = np.split(ds_pred_np_one_img, indices_or_sections=num_patches_per_col, axis=dim_keys_pred["n_samples"])
        split_labels_ds_list = np.split(ds_labels_np_one_img, indices_or_sections=num_patches_per_col, axis=dim_keys_labels["n_samples"])
        split_examples_ds_list = np.split(ds_examples_np_one_img, indices_or_sections=num_patches_per_col, axis=dim_keys_examples["n_samples"])
        split_pred_ds_list_concat = []
        split_labels_ds_list_concat = []
        split_examples_ds_list_concat = []
        for split_pred_ds, split_labels_ds, split_examples_ds in zip(split_pred_ds_list, split_labels_ds_list, split_examples_ds_list):
            split_pred_ds_list_concat.append(np.concatenate(np.split(split_pred_ds, indices_or_sections=split_pred_ds.shape[dim_keys_pred["n_samples"]], axis=dim_keys_pred["n_samples"]), axis=dim_keys_pred["lat_dim"]))
            split_labels_ds_list_concat.append(np.concatenate(np.split(split_labels_ds, indices_or_sections=split_labels_ds.shape[dim_keys_labels["n_samples"]], axis=dim_keys_labels["n_samples"]), axis=dim_keys_labels["lat_dim"]))
            split_examples_ds_list_concat.append(np.concatenate(np.split(split_examples_ds, indices_or_sections=split_examples_ds.shape[dim_keys_examples["n_samples"]], axis=dim_keys_examples["n_samples"]), axis=dim_keys_examples["lat_dim"]))
        complete_pred_ds = np.concatenate(split_pred_ds_list_concat, axis=dim_keys_pred["lon_dim"])
        complete_pred_ds = np.take(complete_pred_ds, 0, axis=dim_keys_pred["n_samples"])
        complete_labels_ds = np.concatenate(split_labels_ds_list_concat, axis=dim_keys_labels["lon_dim"])
        complete_labels_ds = np.take(complete_labels_ds, 0, axis=dim_keys_labels["n_samples"])
        complete_examples_ds = np.concatenate(split_examples_ds_list_concat, axis=dim_keys_examples["lon_dim"])
        complete_examples_ds = np.take(complete_examples_ds, 0, axis=dim_keys_examples["n_samples"])

        # Select our variables for plotting including latlons
        lons = np.take(complete_examples_ds, var_keys_examples["lon"], axis=0)
        lats = np.take(complete_examples_ds, var_keys_examples["lat"], axis=0)
        comp_dz = np.take(complete_examples_ds, var_keys_examples["comp_dz"], axis=0)
        pred_hail = np.take(complete_pred_ds, var_keys_pred["MESH_class_bin_1"], axis=0)
        true_hail = np.take(complete_labels_ds, var_keys_labels["MESH_class_bin_1"], axis=0)

        domain_refl_plot(plot_output_dir, lons, lats, comp_dz, pred_hail, init_datetime, plot_num)
        domain_truth_plot(plot_output_dir, lons, lats, true_hail, pred_hail, init_datetime, plot_num)

        plot_num = plot_num + 1


def domain_refl_plot(plot_output_dir, lons, lats, comp_dz, pred_val, init_datetime, plot_num = 0):
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
    # ax.set_xlim((min(lons.flatten())+1.5,max(lons.flatten())-1.5))
    # ax.set_ylim((min(lats.flatten())+2,max(lats.flatten())))

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
    plt.colorbar(fraction=0.043, pad=0.05, ticks = np.arange(5,76,5)).set_label(label=comp_dz_label,size=15)
    plt.contour(lons,lats,pred_val,np.arange(0,101,10)/100, colors='k', transform=ccrs.PlateCarree(), linewidths=0.6, vmin=0, vmax=1, zorder=11)
    plt.colorbar(fraction=0.043, pad=0.02, ticks = np.arange(0,101,10)/100).set_label(label=pred_val_label,size=15)

    plt.savefig(os.path.join(plot_output_dir, "refl_plot_" + str(plot_num) + '.png'))


def domain_truth_plot(plot_output_dir, lons, lats, true_val, pred_val, init_datetime, plot_num = 0):
    # Titles and labels
    current_datetime = init_datetime + timedelta(minutes=plot_num*5) # *5 for 5 minutes
    init_time_offset_str = "{:02d}".format(plot_num*5)
    title = "Severe Hail Event on " + current_datetime.strftime("%Y-%m-%d:%H%M") + " UTC (Initialization Time + " + init_time_offset_str + " Minutes)"
    real_val_label = "GridRad Location of Hail > 10 mm"
    pred_val_label = "Predicted Probability of Hail > 10 mm"

    # Setup figure
    fig = plt.figure(figsize=(16, 12), dpi=300)
    fig.patch.set_facecolor('white')
    plt.tight_layout()
    ax = plt.axes(projection=ccrs.PlateCarree())
    plt.title(title, fontsize=20)
    # ax.set_xlim((min(lons.flatten())+1.5,max(lons.flatten())-1.5))
    # ax.set_ylim((min(lats.flatten())+2,max(lats.flatten())))

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
    # Add actual plots
    true_val[np.nonzero(true_val == 0)] = np.nan
    plt.contourf(lons,lats,true_val, [0, 1], colors='red', transform=ccrs.PlateCarree(), zorder=10, alpha=0.7)
    cbar = plt.colorbar(fraction=0.043, pad=0.05)
    cbar.set_ticks([])
    cbar.set_label(label=real_val_label,size=15)
    plt.contourf(lons,lats,pred_val,np.arange(10,101,10)/100, cmap="Greens", transform=ccrs.PlateCarree(), vmin=0.1, vmax=1, zorder=11, alpha=0.8)
    plt.colorbar(fraction=0.043, pad=0.02, ticks = np.arange(10,101,10)/100).set_label(label=pred_val_label,size=15)

    plt.savefig(os.path.join(plot_output_dir, "truth_plot_" + str(plot_num) + '.png'))


# NOTE: This is only for binary case
def plot_test_data_plots(args):
    examples_glob = args["processed_examples"]
    labels_glob = args["unprocessed_labels"]
    plot_output_dir = args["plot_output_dir"]
    # predictions_paths = args["predictions_paths"]
    predictions_paths = [["/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour_fixed-128_size/patches/test/predictions/y_hats_00.nc",
                         "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour_fixed-128_size/patches/test/predictions/y_hats_15.nc",
                         "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour_fixed-128_size/patches/test/predictions/y_hats_30.nc",
                         "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour_fixed-128_size/patches/test/predictions/y_hats_45.nc",
                         "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour_fixed-128_size/patches/test/predictions/y_hats_55.nc"]]
    model_names = ["Multi-Model"]
    hail_name = "MESH_class_bin"
    lead_time_names_pretty = ["Init Plus 00", "Init Plus 15", "Init Plus 30", "Init Plus 45", "Init Plus 60"]
    lead_time_minutes = [0, 15, 30, 45, 60]
    lead_time_indices = [0, 3, 6, 9, 11]

    # Load all the xarray dataset objects
    examples_list = glob.glob(examples_glob)
    examples_list.sort()
    examples_dataset = xr.open_mfdataset(examples_list, concat_dim='n_samples', combine='nested', engine='netcdf4')
    labels_list = glob.glob(labels_glob)
    labels_list.sort()
    labels_dataset = xr.open_mfdataset(labels_list, concat_dim='n_samples', combine='nested', engine='netcdf4')

    labels_dataset = labels_dataset.transpose("n_samples", "lat_dim", "lon_dim", "time_dim")

    # Load predictions and truths for each of the valid times
    truths_flattened = []
    predictions_flattened = []
    for predictions_path in predictions_paths:
        predictions = []
        truths = []

        for i, lead_time_index in enumerate(lead_time_indices):
            if type(predictions_path) is list:
                predictions.append(xr.open_dataset(predictions_path[i])[hail_name].to_numpy().ravel())
            else:
                predictions.append(xr.open_dataset(predictions_path)[{"time_dim": lead_time_index}][hail_name].to_numpy().ravel())
            
            truth = unpack_ne_dim_output(labels_dataset[{"time_dim": lead_time_index}][hail_name].to_numpy(), 18, 1).ravel()
            truths.append(truth)
        
        truths_flattened.append(truths)
        predictions_flattened.append(predictions)

    # Load a hailcast list
    hailcast_flattened = []
    for lead_time_index in lead_time_indices:
        hailcast_flattened.append(examples_dataset[{"time_dim": lead_time_index}]['hailcast'].to_numpy().ravel())

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
    for truth, hailcast in zip(truths_flattened[-1], hailcast_flattened):
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
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    for i, max_csi in enumerate(max_csis):
        plt.plot(lead_time_minutes, max_csi, linestyle='-', label=model_names[i], marker='o')
    plt.plot(lead_time_minutes, max_csis_hailcast, linestyle='-', label="Hailcast", marker='o')
    plt.ylabel('Max CSI')
    plt.xlabel('Lead Time (mins)')
    plt.title('Max CSI With Lead Time')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_output_dir, 'max_csi.png'))

    # # Make the ROC plot
    # fig = plt.figure()
    # fig.patch.set_facecolor('white')
    # plt.plot([0,1], [0,1], linestyle='--', label='No Skill: AUC=%.3f' % (0.5))
    # for roc_auc_score_value, roc_auc_score_hailcast_value, roc_curve_value, roc_curve_hailcast_value, lead_time_name in zip(roc_auc_scores, roc_auc_scores_hailcast, roc_curves, roc_curves_hailcast, lead_time_names_pretty):
    #     plt.plot(roc_curve_value[0], roc_curve_value[1], linestyle='-', label="ML Hail " + lead_time_name + ': AUC=%.3f' % (roc_auc_score_value))
    #     plt.plot(roc_curve_hailcast_value[0], roc_curve_hailcast_value[1], linestyle='-.', label="Hailcast " + lead_time_name + ': AUC=%.3f' % (roc_auc_score_hailcast_value), color=plt.gca().lines[-1].get_color())
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic Curve')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(os.path.join(plot_output_dir, 'ROC_Curve.png'))

    # # Make the calibration plot
    # fig = plt.figure()
    # fig.patch.set_facecolor('white')
    # plt.plot([0,1], linestyle='--')
    # for calibration_curve_val, calibration_curve_hailcast_val, lead_time_name in zip(calibration_curves, calibration_curves_hailcast, lead_time_names_pretty):
    #     plt.plot(calibration_curve_val[0], calibration_curve_val[1], linestyle='-', label="ML Hail " + lead_time_name)
    #     plt.plot(calibration_curve_hailcast_val[0], calibration_curve_hailcast_val[1], linestyle='-.', label="Hailcast " + lead_time_name, color=plt.gca().lines[-1].get_color())
    # plt.ylabel("Observed Frequency")
    # plt.xlabel("Predicted Probability")
    # plt.title("Reliability Diagram")
    # plt.legend(loc="best")
    # plt.tight_layout()
    # plt.savefig(os.path.join(plot_output_dir, 'Reliability_Diagram.png'))


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