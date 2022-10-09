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
# TEMP!!!!!!!!
import tensorflow as tf
from custom_metrics import MaxCriticalSuccessIndex

def create_parser():
    '''
    Create argument parser
    '''
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description='Unet Preprocessing', fromfile_prefix_chars='@')

    parser.add_argument('--predictions_path', type=str, default='/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/trained_at_init_time/patches/animations/20190501/predictions/y_hats.nc')
    parser.add_argument('--unprocessed_examples', type=str, default='/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/trained_at_init_time/patches/animations/20190501/unprocessed/examples/*')
    parser.add_argument('--unprocessed_labels', type=str, default='/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/trained_at_init_time/patches/animations/20190501/unprocessed/labels/*')
    parser.add_argument('--test_data_dir', type=str, default='/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/trained_at_init_time/patches/test')
    # parser.add_argument('--truths_path', type=str, default='/Users/tschmidt/repos/hail/data/predictions/no_refl_trained_at_init_time_2022_08_03/y_hats.nc')
    # parser.add_argument('--other_features_path', type=str, default='/Users/tschmidt/repos/hail/data/predictions/no_refl_trained_at_init_time_2022_08_03/y_hats.nc')
    parser.add_argument('--plot_output_dir', type=str, default='/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/trained_at_init_time/images/test_dataset_plots')
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
    # predictions_path = args["predictions_path"]
    test_data_dir = args["test_data_dir"]
    # truths_path = args["truths_path"]
    # class_names = args["class_names"]
    plot_output_dir = args["plot_output_dir"]
    hail_name = "MESH_class_bin"
    lead_time_names = ["init_plus_00", "init_plus_15", "init_plus_30", "init_plus_45", "init_plus_60"]
    lead_time_names_pretty = ["Init Plus 00", "Init Plus 15", "Init Plus 30", "Init Plus 45", "Init Plus 60"]

    # # TEMP!!!!!!
    # derp = glob.glob("/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/trained_at_init_time/patches/test/init_plus_00/unprocessed/labels/*")
    # derp.sort()
    # truth = xr.open_mfdataset(derp, concat_dim='n_samples', combine='nested', engine='netcdf4', decode_cf=False)[hail_name]
    # # truth = truth.transpose("n_samples", ...)
    # truth = truth.to_numpy().ravel()
    # # print(xr.open_mfdataset(derp, concat_dim='n_samples', combine='nested', engine='netcdf4', decode_cf=False))
    # # print(xr.open_dataset("/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/trained_at_init_time/patches/test/init_plus_00/predictions/y_hats_old.nc", decode_cf=False))
    # prediction = xr.open_dataset("/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/trained_at_init_time/patches/test/init_plus_00/predictions/y_hats_old.nc", decode_cf=False)[hail_name].to_numpy().ravel()
    # derp = glob.glob("/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/trained_at_init_time/patches/test/init_plus_00/unprocessed/examples/*")
    # derp.sort()
    # hailcast = xr.open_mfdataset(derp, concat_dim='n_samples', combine='nested', engine='netcdf4', decode_cf=False)['hailcast']
    # hailcast = hailcast.to_numpy().ravel()
    # hailcast_class = np.zeros(len(hailcast), dtype=np.int64)
    # hailcast_class[np.nonzero(hailcast > 0.393701)] = 1
    # hailcast = hailcast / 0.393701
    # hailcast[np.nonzero(hailcast > 1)] = 1
    # # print(np.sum(hailcast_class))
    # # print(np.sum(truth))
    # bin_acc = MaxCriticalSuccessIndex()
    # # bin_acc = tf.keras.metrics.BinaryAccuracy()
    # bin_acc.update_state(truth, prediction)
    # print(bin_acc.result().numpy())
    # bin_acc.reset_state()
    # bin_acc.update_state(truth, hailcast)
    # print(bin_acc.result().numpy())
    # # prediction[np.nonzero(prediction < 0.1)] = 0
    # print(np.max(prediction))
    # # print(np.min(truth))
    # # prediction = np.ones(len(truth))
    # fpr, tpr = calibration_curve(truth, prediction, n_bins=100)
    # plt.plot([0,1], linestyle='--', label="dumb")
    # plt.plot(tpr, fpr, linestyle='-', label="ML Hail init_time_00")
    # fpr, tpr = calibration_curve(truth, hailcast, n_bins=100)
    # plt.plot(tpr, fpr, linestyle='-.', label="Hailcast init_time_00", color=plt.gca().lines[-1].get_color())
    # plt.legend()
    # plt.savefig(os.path.join(plot_output_dir, 'Reliability_Diagram.png'))

    # Load all datasets for each of the valid times (see lead_time_names above)
    examples_datasets = []
    labels_datasets = []
    predictions_datasets = []
    for lead_time_name in lead_time_names:
        examples_list = glob.glob(os.path.join(test_data_dir, lead_time_name + "/processed/examples/*"))
        examples_list.sort()
        examples_datasets.append(xr.open_mfdataset(examples_list, concat_dim='n_samples', combine='nested', engine='netcdf4', decode_cf=False)) #TODO: REMOVE decode_cf when possible
        labels_list = glob.glob(os.path.join(test_data_dir, lead_time_name + "/processed/labels/*"))
        labels_list.sort()
        labels_datasets.append(xr.open_mfdataset(labels_list, concat_dim='n_samples', combine='nested', engine='netcdf4', decode_cf=False)) #TODO: REMOVE decode_cf when possible
        predictions_datasets.append(xr.open_dataset(os.path.join(test_data_dir, lead_time_name + "/predictions/y_hats_old.nc"), decode_cf=False))

    # # Assumes that predictions and truths have the same number of classes (features)
    # predictions = xr.open_dataset(predictions_path, decode_cf=False) #.to_array().to_numpy()
    # # truths = xr.open_dataset(truths_path).to_array().to_numpy()
    # ds_labels = xr.open_mfdataset(unprocessed_labels, concat_dim='n_samples', combine='nested', engine='netcdf4', decode_cf=False) #TODO: REMOVE decode_cf when possible

    # Covert the numpy arrays into pixel-by-pixel comparison
    truths_flattened = []
    predictions_flattened = []
    hailcast_flattened = []
    for example_ds, labels_ds, pred_ds in zip(examples_datasets, labels_datasets, predictions_datasets):
        truths_flattened.append(labels_ds[hail_name].to_numpy().ravel())
        predictions_flattened.append(pred_ds[hail_name].to_numpy().ravel())
        hailcast_flattened.append(example_ds['hailcast'].to_numpy().ravel())
        # hailcast = example_ds["hailcast"].to_numpy().ravel()
        # hailcast = hailcast / 0.393701
        # hailcast[np.nonzero(hailcast > 1)] = 1
        # hailcast_class = np.zeros(len(hailcast), dtype=np.int64)
        # hailcast_class[np.nonzero(hailcast > 0.393701)] = 1
        # hailcast_flattened.append(hailcast) # Was hailcast_class in append
    # for i in range(predictions.shape[0]):
    #     truths_flattened.append(truths[i,...].ravel())
    #     predictions_flattened.append(predictions[i,...].ravel())

    # Calculate various ROC values
    roc_auc_scores = []
    roc_auc_scores_hailcast = []
    roc_curves = []
    roc_curves_hailcast = []
    for truth, prediction, hailcast in zip(truths_flattened, predictions_flattened, hailcast_flattened):
        roc_auc_scores.append(roc_auc_score(truth, prediction))
        roc_auc_scores_hailcast.append(roc_auc_score(truth, hailcast))
        fpr, tpr, thresholds = roc_curve(truth, prediction)
        roc_curves.append((fpr, tpr))
        fpr, tpr, thresholds = roc_curve(truth, hailcast)
        roc_curves_hailcast.append((fpr, tpr))

    # Make the ROC plot
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    plt.plot([0,1], [0,1], linestyle='--', label='No Skill: AUC=%.3f' % (0.5))
    for roc_auc_score_value, roc_auc_score_hailcast_value, roc_curve_value, roc_curve_hailcast_value, lead_time_name in zip(roc_auc_scores, roc_auc_scores_hailcast, roc_curves, roc_curves_hailcast, lead_time_names_pretty):
        plt.plot(roc_curve_value[0], roc_curve_value[1], linestyle='-', label="ML Hail " + lead_time_name + ': AUC=%.3f' % (roc_auc_score_value))
        plt.plot(roc_curve_hailcast_value[0], roc_curve_hailcast_value[1], linestyle='-.', label="Hailcast " + lead_time_name + ': AUC=%.3f' % (roc_auc_score_hailcast_value), color=plt.gca().lines[-1].get_color())
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_output_dir, 'ROC_Curve.png'))

    # Calculate the various calibration values
    calibration_curves = []
    calibration_curves_hailcast = []
    for truth, prediction, hailcast in zip(truths_flattened, predictions_flattened, hailcast_flattened):
        prob_true, prob_pred = calibration_curve(truth, prediction, n_bins=100)
        calibration_curves.append((prob_pred, prob_true))
        prob_true, prob_pred = calibration_curve(truth, hailcast, n_bins=100)
        calibration_curves_hailcast.append((prob_pred, prob_true))

    # Make the calibration plot
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    plt.plot([0,1], linestyle='--')
    for calibration_curve_val, calibration_curve_hailcast_val, lead_time_name in zip(calibration_curves, calibration_curves_hailcast, lead_time_names_pretty):
        plt.plot(calibration_curve_val[0], calibration_curve_val[1], linestyle='-', label="ML Hail " + lead_time_name)
        plt.plot(calibration_curve_hailcast_val[0], calibration_curve_hailcast_val[1], linestyle='-.', label="Hailcast " + lead_time_name, color=plt.gca().lines[-1].get_color())
    plt.ylabel("Observed Frequency")
    plt.xlabel("Predicted Probability")
    plt.title("Reliability Diagram")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_output_dir, 'Reliability_Diagram.png'))


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