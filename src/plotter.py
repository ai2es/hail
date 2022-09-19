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

def create_parser():
    '''
    Create argument parser
    '''
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description='Unet Preprocessing', fromfile_prefix_chars='@')

    parser.add_argument('--predictions_path', type=str, default='/Users/tschmidt/repos/hail/data/predictions/no_refl_trained_at_init_time_2022_08_03/y_hats.nc')
    parser.add_argument('--truths_path', type=str, default='/Users/tschmidt/repos/hail/data/predictions/no_refl_trained_at_init_time_2022_08_03/y_hats.nc')
    parser.add_argument('--other_features_path', type=str, default='/Users/tschmidt/repos/hail/data/predictions/no_refl_trained_at_init_time_2022_08_03/y_hats.nc')
    parser.add_argument('--plot_output_dir', type=str, default='/Users/tschmidt/repos/hail/data/predictions/no_refl_trained_at_init_time_2022_08_03/y_hats.nc')
    parser.add_argument('--class_names', type=str, nargs='+', default=['Hail'])

    return parser


def plot_casestudy_plots(args):
    num_patches_per_col = 4 # TODO: Make this a function of patch_size and orig domain size?
    ds_path = "/Users/tschmidt/repos/hail/data/predictions/20190501_casestudy/predictions3.nc"

    ds = xr.open_dataset(ds_path, decode_cf=False) #TODO: REMOVE decode_cf when possible

    var_keys = {key_str:i for i,key_str in enumerate(ds.keys())}
    dim_keys = {key_str:i+1 for i,key_str in enumerate(list(ds.dims))}
    ds_np = ds.to_array().to_numpy()

    # Stitch together all patches from inside the netcdf file.
    # Assumes that single netcdf file is for one whole domain and that the patches are in order
    split_ds_list = np.split(ds_np, indices_or_sections=num_patches_per_col, axis=dim_keys["n_samples"])
    split_ds_list_concat = []
    for split_ds in split_ds_list:
        split_ds_list_concat.append(np.concatenate(np.split(split_ds, indices_or_sections=split_ds.shape[dim_keys["n_samples"]], axis=dim_keys["n_samples"]), axis=dim_keys["lat_dim"]))
    complete_ds = np.concatenate(split_ds_list_concat, axis=dim_keys["lon_dim"])
    complete_ds = np.take(complete_ds, 0, axis=dim_keys["n_samples"])

    # Select our variables for plotting including latlons
    lons = np.take(complete_ds, var_keys["lon"], axis=0)
    lats = np.take(complete_ds, var_keys["lat"], axis=0)
    comp_dz = np.take(complete_ds, var_keys["comp_dz"], axis=0)
    pred_hail = np.take(complete_ds, var_keys["predicted_hail"], axis=0)

    title = "May 1st 2019 Severe Hail Event 20:00 UTC (Init time + 60 mins)"
    comp_dz_label = "WoFS Composite Reflectivity (dBZ > 5)"
    pred_val_label = "Probability of Hail > 10 mm"


def domain_refl_plot(ds_path, plot_output_dir, num_patches_per_col, lons, lats, comp_dz, pred_val, comp_dz_label, pred_val_label, title, plot_num = 0):
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

    plt.savefig(os.path.join(plot_output_dir, str(plot_num) + '.png'))


def plot_test_data_plots(args):
    predictions_path = args["predictions_path"]
    truths_path = args["truths_path"]
    class_names = args["class_names"]
    plot_output_dir = args["plot_output_dir"]

    # Assumes that predictions and truths have the same number of classes (features)
    predictions = xr.open_dataset(predictions_path).to_array().to_numpy()
    truths = xr.open_dataset(truths_path).to_array().to_numpy()

    # Covert the numpy arrays into pixel-by-pixel comparison
    truths_flattened = []
    predictions_flattened = []
    for i in range(predictions.shape[0]):
        truths_flattened.append(truths[i,...].ravel())
        predictions_flattened.append(predictions[i,...].ravel())

    # Calculate various ROC values
    roc_auc_scores = []
    roc_curves = []
    for truth, prediction in zip(truths_flattened, predictions_flattened):
        roc_auc_scores.append(roc_auc_score(truth, prediction))
        fpr, tpr, thresholds = roc_curve(truth, prediction)
        roc_curves.append((fpr, tpr))

    # Make the ROC plot
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    plt.plot([0,1], [0,1], linestyle='--', label='No Skill: AUC=%.3f' % (0.5))
    for i, (roc_auc_score_value, roc_curve_value) in enumerate(zip(roc_auc_scores, roc_curves)):
        plt.plot(roc_curve_value[0], roc_curve_value[1], linestyle='-', label=class_names[i] + ': AUC=%.3f' % (roc_auc_score_value))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_output_dir, 'ROC_Curve.png'))

    # Calculate the various calibration values
    calibration_curves = []
    for truth, prediction in zip(truths_flattened, predictions_flattened):
        prob_true, prob_pred = calibration_curve(truth, prediction, n_bins=100)
        calibration_curves.append((prob_pred, prob_true))

    # Make the calibration plot
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    plt.plot([0,1], linestyle='--')
    for i, calibration_curve_val in enumerate(calibration_curves):
        plt.plot(calibration_curve_val[0], calibration_curve_val[1], linestyle='-', label=class_names[i])
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

    plot_test_data_plots(args)