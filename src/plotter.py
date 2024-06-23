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
from custom_metrics import MaxCriticalSuccessIndex, neighborhood_max_csi
from preprocessor import unpack_ne_dim_output
from storm_report_loader import StormReportLoader
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from sklearn.metrics import precision_recall_curve
import warnings
import matplotlib as mpl
import shapely.geometry
from descartes import PolygonPatch
from scipy.ndimage import gaussian_filter, maximum_filter


##### BEGIN CODE BORROWED FROM FLORA #####


def _vertex_arrays_to_list(vertex_x_coords, vertex_y_coords):
    """Converts vertices of simple polygon from two arrays to one list.
    x- and y-coordinates may be in one of three formats (see docstring at top of
    file).
    V = number of vertices
    :param vertex_x_coords: See documentation for _check_vertex_arrays.
    :param vertex_y_coords: See documentation for _check_vertex_arrays.
    :return: vertex_coords_as_list: length-V list, where each element is an
        (x, y) tuple.
    """
    num_vertices = len(vertex_x_coords)
    vertex_coords_as_list = []
    for i in range(num_vertices):
        vertex_coords_as_list.append((vertex_x_coords[i], vertex_y_coords[i]))

    return np.array(vertex_coords_as_list)


def _confidence_interval_to_polygon(
    x_coords_bottom,
    y_coords_bottom,
    x_coords_top,
    y_coords_top
):
    """Generates polygon for confidence interval.
    P = number of points in bottom curve = number of points in top curve
    :param x_coords_bottom: length-P np with x-coordinates of bottom curve
        (lower end of confidence interval).
    :param y_coords_bottom: Same but for y-coordinates.
    :param x_coords_top: length-P np with x-coordinates of top curve (upper
        end of confidence interval).
    :param y_coords_top: Same but for y-coordinates.
    :return: polygon_object: Instance of `shapely.geometry.Polygon`.
    """

    nan_flags_top = np.logical_or(np.isnan(x_coords_top), np.isnan(y_coords_top))
    if np.all(nan_flags_top):
        return None

    nan_flags_bottom = np.logical_or(
        np.isnan(x_coords_bottom), np.isnan(y_coords_bottom)
    )
    if np.all(nan_flags_bottom):
        return None

    real_indices_top = np.where(np.invert(nan_flags_top))[0]
    real_indices_bottom = np.where(np.invert(nan_flags_bottom))[0]

    x_coords_top = x_coords_top[real_indices_top]
    sort_indices_top = np.argsort(-x_coords_top)
    x_coords_top = x_coords_top[sort_indices_top]
    y_coords_top = y_coords_top[real_indices_top][sort_indices_top]

    x_coords_bottom = x_coords_bottom[real_indices_bottom]
    sort_indices_bottom = np.argsort(x_coords_bottom)
    x_coords_bottom = x_coords_bottom[sort_indices_bottom]
    y_coords_bottom = y_coords_bottom[real_indices_bottom][sort_indices_bottom]

    polygon_x_coords = np.concatenate(
        (x_coords_top, x_coords_bottom, np.array([x_coords_top[0]]))
    )
    polygon_y_coords = np.concatenate(
        (y_coords_top, y_coords_bottom, np.array([y_coords_top[0]]))
    )

    return vertex_arrays_to_polygon_object(polygon_x_coords, polygon_y_coords)


def vertex_arrays_to_polygon_object(
    exterior_x_coords,
    exterior_y_coords,
    hole_x_coords_list=None,
    hole_y_coords_list=None,
):
    """Converts polygon from vertex arrays to `shapely.geometry.Polygon` object.
    V_e = number of exterior vertices
    H = number of holes
    V_hi = number of vertices in [i]th hole
    :param exterior_x_coords: np array (length V_e) with x-coordinates of
        exterior vertices.
    :param exterior_y_coords: np array (length V_e) with y-coordinates of
        exterior vertices.
    :param hole_x_coords_list: length-H list, where the [i]th item is a np
        array (length V_hi) with x-coordinates of interior vertices.
    :param hole_y_coords_list: Same as above, except for y-coordinates.
    :return: polygon_object: `shapely.geometry.Polygon` object.
    :raises: ValueError: if the polygon is invalid.
    """

    exterior_coords_as_list = _vertex_arrays_to_list(
        exterior_x_coords, exterior_y_coords
    )
    if hole_x_coords_list is None:
        return shapely.geometry.Polygon(shell=exterior_coords_as_list)

    num_holes = len(hole_x_coords_list)
    outer_list_of_hole_coords = []
    for i in range(num_holes):
        outer_list_of_hole_coords.append(
            _vertex_arrays_to_list(hole_x_coords_list[i], hole_y_coords_list[i])
        )

    polygon_object = shapely.geometry.Polygon(
        shell=exterior_coords_as_list, holes=tuple(outer_list_of_hole_coords)
    )

    if not polygon_object.is_valid:
        raise ValueError("Resulting polygon is invalid.")

    return polygon_object


# My version of Monte's code
def make_confidence_intervals(lead_time_minutes, pred_data, label_data, line_color, n_boot, boot_size, nbhr=False, seed=42):
    num_timesteps = pred_data.shape[-1]
    bootstrapped_csis = np.zeros((num_timesteps, n_boot), dtype=np.float64)

    ds_size = np.minimum(label_data.shape[0], pred_data.shape[0])

    base_random_state = np.random.RandomState(seed)

    for i in range(n_boot):
        idx = base_random_state.choice(ds_size, size=boot_size, replace=True)
        for j in range(num_timesteps):
            single_pred_data = pred_data[...,j]
            single_label_data = label_data[...,j]

            if len(single_pred_data.shape) < len(single_label_data.shape) and len(single_label_data.shape) == 3:
                single_pred_data = single_pred_data.reshape(single_label_data.shape)
            if len(single_label_data.shape) < len(single_pred_data.shape) and len(single_pred_data.shape) == 3:
                single_label_data = single_label_data.reshape(single_pred_data.shape)

            single_pred_data = single_pred_data[idx,...]
            single_label_data = single_label_data[idx,...]

            if nbhr:
                single_pred_data = np.expand_dims(single_pred_data, -1)
                single_label_data = np.expand_dims(single_label_data, -1)
                bootstrapped_csis[j][i] = neighborhood_max_csi(single_label_data, single_pred_data)

            else:
                single_pred_data = single_pred_data.flatten()
                single_label_data = single_label_data.flatten()
                max_csi_metric = MaxCriticalSuccessIndex()
                max_csi_metric.reset_state()
                max_csi_metric.update_state(single_label_data, single_pred_data)
                bootstrapped_csis[j][i] = max_csi_metric.result().numpy()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        y_coords_bottom, y_coords_top = np.nanpercentile(bootstrapped_csis, (2.5, 97.5), axis=1)

    polygon_object = _confidence_interval_to_polygon(
        lead_time_minutes,
        y_coords_bottom,
        lead_time_minutes,
        y_coords_top
    )   

    polygon_colour = mpl.colors.to_rgba(line_color, 0.4)

    polygon_patch = PolygonPatch(
        polygon_object, lw=0, ec=polygon_colour, fc=polygon_colour
    )

    return polygon_patch

##### END CODE BORROWED FROM FLORA #####


def create_parser():
    '''
    Create argument parser
    '''
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description='Unet Preprocessing', fromfile_prefix_chars='@')

    ######## CASESTUDY SETTINGS #####################
    parser.add_argument('--make_static', '-s', action='store_true')
    parser.add_argument('--init_datetime', type=str, default='2017-05-18:1900') # Was '2017-05-18:1900' or '2019-05-28:2230'
    parser.add_argument('--ens_member', type=int, default=1)
    parser.add_argument('--make_hailcast_probabilistic', '-p', action='store_true')
    parser.add_argument('--ens_size', type=int, default=18)
    parser.add_argument('--num_patches_per_col', type=int, default=3)
    parser.add_argument('--num_patches_per_row', type=int, default=3)
    parser.add_argument('--unprocessed_examples', type=str, default='/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/animations/20170518/unprocessed/examples/0000.nc')
    parser.add_argument('--hail_key', type=str, default='MESH_class_bin_severe')
    parser.add_argument('--domain_padding_size', type=int, default=32)
    parser.add_argument('--stitched_domain_root', type=str, default='/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/animations/averaged_anims/20170518/stitched_domain')
    parser.add_argument('--stitched_domain_file', type=str, default='/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/animations/averaged_anims/20170518/stitched_domain/0.nc')
    parser.add_argument('--save_stitched_domain', '-d', action='store_true')
    parser.add_argument('--plot_stitched_domain', '-l', action='store_true')
    ###### BOTH SETTINGS ##############
    parser.add_argument('--plot_animation', '-a', action='store_true')
    parser.add_argument('--include_reports', '-i', action='store_true')
    parser.add_argument('--plot_output_dir', type=str, default='/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/images/animations')
    parser.add_argument('--predictions_paths', type=str, default='/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/animations/20170518/predictions/y_hats.nc')
    # The below setting needs to be glob for graphs and direct path for animations
    parser.add_argument('--unprocessed_labels', type=str, default='/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/animations/20170518/unprocessed/labels/0000.nc')
    ###### TEST DATA PLOTS SETTINGS #############
    parser.add_argument('--hailcast_files', type=str, default='/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour-128_size-more_fields/patches/animations/20190520/unprocessed/examples/0000.nc')

    return parser


def plot_static_casestudies(all_lons, all_lats, all_comp_dz, all_pred_hail, all_true_hail, all_hailcast, init_datetime, include_reports, plot_output_dir):
    lons = all_lons[0]
    lats = all_lats[0]

    pred_val = np.stack(all_pred_hail, axis=-1)[...,3:]
    true_val = np.stack(all_true_hail, axis=-1)[...,3:]
    hailcast = np.stack(all_hailcast, axis=-1)[...,3:]

    init_datetime = init_datetime+timedelta(minutes=15)
    end_datetime = init_datetime+timedelta(minutes=40)

    pred_val = np.max(pred_val, axis=-1)
    true_val = np.max(true_val, axis=-1)
    hailcast = np.max(hailcast, axis=-1)

    # Titles and labels
    title = "Forecasted Severe Hail Swath from " + init_datetime.strftime("%Y-%m-%d:%H%M") + " UTC to " + end_datetime.strftime("%Y-%m-%d:%H%M") + " UTC"
    pred_val_label = "ML Predicted Probability of Hail > 1 inch"

    hailcast_bin = np.zeros(hailcast.shape)
    hailcast_bin[hailcast > 1] = 1 # was 0.393701
    hailcast = hailcast_bin
    hailcast_and_true = np.logical_and(hailcast, true_val).astype(float)
    hailcast[np.nonzero(hailcast == 0)] = np.nan

    # Setup figure
    fig = plt.figure(figsize=(16, 12), dpi=300)
    fig.patch.set_facecolor('white')
    plt.tight_layout()
    ax = plt.axes(projection=ccrs.PlateCarree())
    plt.title(title, fontsize=20)

    # Case study 1
    ax.set_xlim((-102.5, -98))
    ax.set_ylim((33.25, 37))
    # Case study 2
    # ax.set_xlim((-100, -95))
    # ax.set_ylim((34.25, 38))
    # Case study 3
    # ax.set_xlim((-99, -95.5))
    # ax.set_ylim((33.5, 37))
    # Case study 3.5
    # ax.set_xlim((-100.5, -93.5))
    # ax.set_ylim((37, 41.5))

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
    # Used to be stamen but had to change to google
    stamen_terrain = cimgt.GoogleTiles(desired_tile_form='L', style="terrain")
    cmap = plt.get_cmap('gist_yarg')
    new_cmap = colors.LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=0, b=0.3), cmap(np.linspace(0, 0.3, 100)))
    ax.add_image(stamen_terrain, 11, cmap=new_cmap)

    # Add actual plots
    true_val[np.nonzero(true_val == 0)] = np.nan
    hailcast_and_true[np.nonzero(hailcast_and_true == 0)] = np.nan
    plt.contourf(lons,lats,true_val, [0, 1], colors='red', transform=ccrs.PlateCarree(), zorder=10, alpha=1.0)
    plt.contourf(lons,lats,hailcast, [0, 1], colors='blue', transform=ccrs.PlateCarree(), zorder=11, alpha=1.0)
    plt.contourf(lons,lats,hailcast_and_true, [0, 1], colors='orange', transform=ccrs.PlateCarree(), zorder=12, alpha=1.0)
    plt.contour(lons,lats,pred_val,np.arange(0,101,10)/100, colors='k', transform=ccrs.PlateCarree(), linewidths=2.3, vmin=0, vmax=1.0, zorder=13)
    plt.colorbar(fraction=0.043, pad=0.02, ticks = np.arange(0,101,10)/100).set_label(label=pred_val_label,size=30)


    if include_reports:
        loader = StormReportLoader("/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/storm_reports/STORM_EVENTS_2017-2021.csv", "NOAA", init_datetime.strftime("%Y%m%d%H%M"), 40, 0)

        storm_reports_hail_lats, storm_reports_hail_lons = loader.get_hail_reports()
        storm_reports_hail_sig_lats, storm_reports_hail_sig_lons = loader.get_hail_reports('sig_severe')

        plt.scatter(storm_reports_hail_lons, storm_reports_hail_lats, color='k', s=180, marker='o', edgecolors= "white", transform=ccrs.PlateCarree(), zorder=13, label="Severe Hail")
        plt.scatter(storm_reports_hail_sig_lons, storm_reports_hail_sig_lats, color='k', s=700, marker='*', edgecolors= "white", transform=ccrs.PlateCarree(), zorder=14, label="Sig Severe Hail")
        plt.legend(prop={'size': 25}, loc=4).set_zorder(99)

    plt.savefig(os.path.join(plot_output_dir, "swath_plot.png"), bbox_inches='tight')


def plot_averaged_casestudy_plots(args):
    make_hailcast_probabilistic = args["make_hailcast_probabilistic"]
    init_datetime = args["init_datetime"]
    include_reports = args["include_reports"]
    plot_output_dir = args["plot_output_dir"]
    stitched_domain_root = args["stitched_domain_root"]
    domain_padding_size = args["domain_padding_size"]

    init_datetime = datetime.strptime(init_datetime, "%Y-%m-%d:%H%M")

    complete_array, array_keys = stitched_domain_loader(stitched_domain_root, domain_padding_size)

    for time_index in range(12): # number of time steps
        lons = np.take(complete_array[...,time_index], array_keys["lon"], axis=0)
        lats = np.take(complete_array[...,time_index], array_keys["lat"], axis=0)
        comp_dz = np.take(complete_array[...,time_index], array_keys["comp_dz"], axis=0)
        pred_hail = np.take(complete_array[...,time_index], array_keys["pred_hail"], axis=0)
        true_hail = np.take(complete_array[...,time_index], array_keys["true_hail"], axis=0)
        hailcast = np.take(complete_array[...,time_index], array_keys["hailcast"], axis=0)

        domain_refl_plot(plot_output_dir, lons, lats, comp_dz, pred_hail, init_datetime, include_reports, time_index)
        domain_truth_plot(plot_output_dir, lons, lats, true_hail, pred_hail, hailcast, init_datetime, include_reports, make_hailcast_probabilistic, time_index)


# NOTE: Assumes first domain is top left, second is top right,
# third is bottom left, and fourth is bottom right
def stitched_domain_loader(stitched_dir_path, padding_size):
    ds1 = xr.open_dataset(os.path.join(stitched_dir_path, "0.nc"))
    ds2 = xr.open_dataset(os.path.join(stitched_dir_path, "1.nc"))
    ds3 = xr.open_dataset(os.path.join(stitched_dir_path, "2.nc"))
    ds4 = xr.open_dataset(os.path.join(stitched_dir_path, "3.nc"))

    keys_for_numpy = {key_str:i for i,key_str in enumerate(ds1.keys())}

    ds1 = ds1.to_array().to_numpy()
    ds2 = ds2.to_array().to_numpy()
    ds3 = ds3.to_array().to_numpy()
    ds4 = ds4.to_array().to_numpy()

    horizontal_nan_block_shape = list(ds1.shape)
    vertical_nan_block_shape = list(ds1.shape)

    horizontal_nan_block_shape[1] = padding_size
    horizontal_nan_block_shape[2] = horizontal_nan_block_shape[2] + padding_size
    vertical_nan_block_shape[2] = padding_size

    horizontal_nan_block = np.zeros(tuple(horizontal_nan_block_shape))
    horizontal_nan_block[:] = np.nan
    vertical_nan_block = np.zeros(tuple(vertical_nan_block_shape))
    vertical_nan_block[:] = np.nan

    ds1 = np.concatenate((ds1, vertical_nan_block), axis=2)
    ds1 = np.concatenate((ds1, horizontal_nan_block), axis=1)

    ds2 = np.concatenate((vertical_nan_block, ds2), axis=2)
    ds2 = np.concatenate((ds2, horizontal_nan_block), axis=1)

    ds3 = np.concatenate((ds3, vertical_nan_block), axis=2)
    ds3 = np.concatenate((horizontal_nan_block, ds3), axis=1)

    ds4 = np.concatenate((vertical_nan_block, ds4), axis=2)
    ds4 = np.concatenate((horizontal_nan_block, ds4), axis=1)

    complete_array = np.stack([ds1, ds2, ds3, ds4], axis=-1)
    complete_array = np.nanmean(complete_array, axis=-1)
    pred_hail = complete_array[keys_for_numpy["pred_hail"], ...]
    pred_hail_all_timesteps = []
    for i in range(12): # Number of timesteps
        pred_hail_single = pred_hail[...,i]
        pred_hail_single = gaussian_filter(pred_hail_single, sigma=2, truncate=0.5)
        pred_hail_all_timesteps.append(pred_hail_single)
    pred_hail_all_timesteps = np.stack(pred_hail_all_timesteps, axis=-1)
    complete_array[keys_for_numpy["pred_hail"], ...] = pred_hail_all_timesteps

    return complete_array, keys_for_numpy


def plot_casestudy_plots(args):
    predictions_paths = args["predictions_paths"]
    unprocessed_labels_path = args["unprocessed_labels"]
    unprocessed_examples_path = args["unprocessed_examples"]
    ens_member = args["ens_member"]
    make_hailcast_probabilistic = args["make_hailcast_probabilistic"]
    ens_size = args["ens_size"]
    num_patches_per_col = args["num_patches_per_col"]
    num_patches_per_row = args["num_patches_per_row"]
    plot_output_dir = args["plot_output_dir"]
    init_datetime = args["init_datetime"]
    include_reports = args["include_reports"]
    make_static = args["make_static"]
    hail_key = args["hail_key"]
    save_stitched_domain = args["save_stitched_domain"]
    stitched_domain_file = args["stitched_domain_file"]

    init_datetime = datetime.strptime(init_datetime, "%Y-%m-%d:%H%M")

    # NOTE: # This block had to keep decode=False for legacy reasons. So the latlons could be put directly into the numpy arrays without losing them.
    if type(predictions_paths) is list:
        ds_pred = xr.open_mfdataset(predictions_paths, concat_dim='time_dim', combine='nested', engine='netcdf4', decode_cf=False)
    else:
        ds_pred = xr.open_dataset(predictions_paths, decode_cf=False)
    # The following 3 files are assumed to be single files rather than collections of files since this is for one case study
    ds_labels = xr.open_dataset(unprocessed_labels_path, decode_cf=False)
    ds_examples = xr.open_dataset(unprocessed_examples_path, decode_cf=False)
    ds_hailcast = xr.open_dataset(unprocessed_examples_path, decode_cf=False)

    ds_examples = ds_examples[{"ne": ens_member}]
    ds_examples = ds_examples.drop("time")
    if not make_hailcast_probabilistic:
        ds_hailcast = ds_hailcast[{"ne": ens_member}]
    ds_hailcast = ds_hailcast.drop("time")

    ds_pred = ds_pred.transpose("n_samples", "lat_dim", "lon_dim", "time_dim")
    ds_labels = ds_labels.transpose("n_samples", "lat_dim", "lon_dim", "time_dim")
    ds_examples = ds_examples.transpose("n_samples", "lat_dim", "lon_dim", "time_dim")
    if make_hailcast_probabilistic:
        ds_hailcast = ds_hailcast.transpose("n_samples", "lat_dim", "lon_dim", "time_dim", "ne")
    else:
        ds_hailcast = ds_hailcast.transpose("n_samples", "lat_dim", "lon_dim", "time_dim")

    # Because we are doing array splitting we need to convert everything to numpy arrays and therefore need to keep track of our keys and dims
    var_keys_pred = {key_str:i for i,key_str in enumerate(ds_pred.keys())}
    var_keys_labels = {key_str:i for i,key_str in enumerate(ds_labels.keys())}
    var_keys_examples = {key_str:i for i,key_str in enumerate(ds_examples.keys())}
    var_keys_hailcast = {key_str:i for i,key_str in enumerate(ds_hailcast.keys())}

    # corner_patch_index = ens_member * num_patches_per_col * num_patches_per_row

    all_lons = []
    all_lats = []
    all_comp_dz = []
    all_pred_hail = []
    all_true_hail = []
    all_hailcast = []

    for time_index in range(12): # number of time steps
        selected_patches = np.arange(ens_member, ens_size * num_patches_per_col * num_patches_per_row + ens_member, ens_size)

        ds_pred_np = ds_pred[{"time_dim": time_index}].to_array().to_numpy()
        ds_labels_np = ds_labels[{"time_dim": time_index}].to_array().to_numpy()
        ds_examples_np = ds_examples[{"time_dim": time_index}].to_array().to_numpy()
        ds_hailcast_np = ds_hailcast[{"time_dim": time_index}].to_array().to_numpy()

        if make_hailcast_probabilistic:
            hailcast_class = np.zeros(ds_hailcast_np.shape, dtype=np.float64)
            hailcast_class[np.nonzero(ds_hailcast_np > 1)] = 1
            ds_hailcast_np = np.sum(hailcast_class, axis=-1)/ens_size

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
        comp_dz = np.take(complete_examples_ds, var_keys_examples["ZH"], axis=0) # ZH for gridrad comp_dz for WoFS
        pred_hail = np.take(complete_pred_ds, var_keys_pred[hail_key], axis=0)
        true_hail = np.take(complete_labels_ds, var_keys_labels[hail_key], axis=0)
        hailcast = np.take(complete_hailcast_ds, var_keys_hailcast["hailcast"], axis=0)

        if make_static or save_stitched_domain:
            all_lons.append(lons)
            all_lats.append(lats)
            all_comp_dz.append(comp_dz)
            all_pred_hail.append(pred_hail)
            all_true_hail.append(true_hail)
            all_hailcast.append(hailcast)

        elif not save_stitched_domain:
            domain_refl_plot(plot_output_dir, lons, lats, comp_dz, pred_hail, init_datetime, include_reports, time_index)
            domain_truth_plot(plot_output_dir, lons, lats, true_hail, pred_hail, hailcast, init_datetime, include_reports, make_hailcast_probabilistic, time_index)

    if make_static:
        plot_static_casestudies(all_lons, all_lats, all_comp_dz, all_pred_hail, all_true_hail, all_hailcast, init_datetime, include_reports, plot_output_dir)

    elif save_stitched_domain:
        all_lons = np.stack(all_lons, axis=-1)
        all_lats = np.stack(all_lats, axis=-1)
        all_comp_dz = np.stack(all_comp_dz, axis=-1)
        all_pred_hail = np.stack(all_pred_hail, axis=-1)
        all_true_hail = np.stack(all_true_hail, axis=-1)
        all_hailcast = np.stack(all_hailcast, axis=-1)

        new_ds_dict = {"lon": (["lat_dim", "lon_dim", "time_dim"], all_lons),
                       "lat": (["lat_dim", "lon_dim", "time_dim"], all_lats),
                       "comp_dz": (["lat_dim", "lon_dim", "time_dim"], all_comp_dz),
                       "pred_hail": (["lat_dim", "lon_dim", "time_dim"], all_pred_hail),
                       "true_hail": (["lat_dim", "lon_dim", "time_dim"], all_true_hail),
                       "hailcast": (["lat_dim", "lon_dim", "time_dim"], all_hailcast)}

        ds_return = xr.Dataset(data_vars = new_ds_dict)

        ds_return.to_netcdf(stitched_domain_file)


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
    ml_time_offset_str = "{:02d}".format(plot_num*5 - 15)

    # Setup figure
    fig = plt.figure(figsize=(16, 12), dpi=300)
    fig.patch.set_facecolor('white')
    plt.tight_layout()
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Case study 1
    ax.set_xlim((-102.5, -98))
    ax.set_ylim((33.25, 37))
    # Case study 2
    # ax.set_xlim((-100, -95))
    # ax.set_ylim((34.25, 38))
    # Case study 3
    # ax.set_xlim((-99, -95.5))
    # ax.set_ylim((33.5, 37))
    # Case study 3.5
    # ax.set_xlim((-100.5, -93.5))
    # ax.set_ylim((37, 41.5))

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
    # Used to be stamen but had to change to google
    stamen_terrain = cimgt.GoogleTiles(desired_tile_form='L', style="terrain")
    cmap = plt.get_cmap('gist_yarg')
    new_cmap = colors.LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=0, b=0.3), cmap(np.linspace(0, 0.3, 100)))
    ax.add_image(stamen_terrain, 11, cmap=new_cmap)

    # Add actual plots
    plt.contourf(lons,lats,comp_dz,np.arange(5, 76, 1), vmin = 5, vmax = 75, cmap=my_cmap, transform=ccrs.PlateCarree(), zorder=10)
    pred_val[pred_val < 0.02] = np.nan
    plt.contour(lons,lats,pred_val,np.arange(0,61,6)/100, colors='k', transform=ccrs.PlateCarree(), linewidths=2.3, vmin=0, vmax=0.6, zorder=11)

    if include_reports:
        current_datetime = current_datetime-timedelta(minutes=10)
        loader = StormReportLoader("/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/storm_reports/STORM_EVENTS_2017-2021.csv", "NOAA", current_datetime.strftime("%Y%m%d%H%M"), 20, 0)

        storm_reports_hail_lats, storm_reports_hail_lons = loader.get_hail_reports()
        storm_reports_hail_sig_lats, storm_reports_hail_sig_lons = loader.get_hail_reports('sig_severe')

        plt.scatter(storm_reports_hail_lons, storm_reports_hail_lats, color='k', s=180, marker='o', edgecolors= "white", transform=ccrs.PlateCarree(), zorder=12, label="Severe Hail")
        plt.scatter(storm_reports_hail_sig_lons, storm_reports_hail_sig_lats, color='k', s=700, marker='*', edgecolors= "white", transform=ccrs.PlateCarree(), zorder=13, label="Sig Severe Hail")

    plt.savefig(os.path.join(plot_output_dir, "refl_plot_" + "{:04d}".format(plot_num) + '.png'), bbox_inches='tight')


def domain_truth_plot(plot_output_dir, lons, lats, true_val, pred_val, hailcast, init_datetime, include_reports, make_hailcast_probabilistic, plot_num = 0):
    # Titles and labels
    current_datetime = init_datetime + timedelta(minutes=plot_num*5) # *5 for 5 minutes
    init_time_offset_str = "{:02d}".format(plot_num*5)
    ml_time_offset_str = "{:02d}".format(plot_num*5 - 15)

    ###### printed CSI values ########
    print("----------------")
    print("Timestep: " + str(plot_num))

    lon_min = -102.5
    lon_max = -98
    lat_min = 33.25
    lat_max = 37

    lon_min_index = np.abs(lons[0,:] - lon_min).argmin()
    lon_max_index = np.abs(lons[0,:] - lon_max).argmin()
    lat_min_index = np.abs(lats[:,0] - lat_min).argmin()
    lat_max_index = np.abs(lats[:,0] - lat_max).argmin()

    true_sliced = true_val[lat_min_index:lat_max_index+1, lon_min_index:lon_max_index+1]
    pred_sliced = pred_val[lat_min_index:lat_max_index+1, lon_min_index:lon_max_index+1]
    hailcast_sliced = hailcast[lat_min_index:lat_max_index+1, lon_min_index:lon_max_index+1]

    true_sliced[np.isnan(true_sliced)] = 0
    pred_sliced[np.isnan(pred_sliced)] = 0
    hailcast_sliced[np.isnan(hailcast_sliced)] = 0

    max_csi_metric = MaxCriticalSuccessIndex()
    max_csi_metric.reset_state()
    max_csi_metric.update_state(true_sliced.flatten(), hailcast_sliced.flatten())
    print("Hailcast Max CSI: " + str(max_csi_metric.result().numpy()))

    max_csi_metric = MaxCriticalSuccessIndex()
    max_csi_metric.reset_state()
    max_csi_metric.update_state(true_sliced.flatten(), pred_sliced.flatten())
    print("U-Net Max CSI: " + str(max_csi_metric.result().numpy()))

    true_sliced = np.expand_dims(true_sliced, -1)
    pred_sliced = np.expand_dims(pred_sliced, -1)
    hailcast_sliced = np.expand_dims(hailcast_sliced, -1)
    true_sliced = np.expand_dims(true_sliced, 0)
    pred_sliced = np.expand_dims(pred_sliced, 0)
    hailcast_sliced = np.expand_dims(hailcast_sliced, 0)

    print("Hailcast Neighborhood Max CSI: " + str(neighborhood_max_csi(true_sliced, hailcast_sliced)))
    print("U-Net Neighborhood Max CSI: " + str(neighborhood_max_csi(true_sliced, pred_sliced)))
    ##########################################

    if make_hailcast_probabilistic:
        hailcast[np.nonzero(hailcast < 0.0001)] = np.nan
    else:
        hailcast_bin = np.zeros(hailcast.shape)
        hailcast_bin[hailcast > 1] = 1 # was 0.393701
        hailcast = hailcast_bin
        hailcast_and_true = np.logical_and(hailcast, true_val).astype(float)
        hailcast[np.nonzero(hailcast == 0)] = np.nan

    # Setup figure
    fig = plt.figure(figsize=(16, 12), dpi=300)
    fig.patch.set_facecolor('white')
    plt.tight_layout()
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Case study 1
    ax.set_xlim((-102.5, -98))
    ax.set_ylim((33.25, 37))
    # Case study 2
    # ax.set_xlim((-100, -95))
    # ax.set_ylim((34.25, 38))
    # Case study 3
    # ax.set_xlim((-99, -95.5))
    # ax.set_ylim((33.5, 37))
    # Case study 3.5
    # ax.set_xlim((-100.5, -93.5))
    # ax.set_ylim((37, 41.5))

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
    # Used to be stamen but had to change to google
    stamen_terrain = cimgt.GoogleTiles(desired_tile_form='L', style="terrain")
    cmap = plt.get_cmap('gist_yarg')
    new_cmap = colors.LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=0, b=0.3), cmap(np.linspace(0, 0.3, 100)))
    ax.add_image(stamen_terrain, 11, cmap=new_cmap)

    # Add actual plots
    true_val[np.nonzero(true_val == 0)] = np.nan
    plt.contourf(lons,lats,true_val, [0, 1], colors='red', transform=ccrs.PlateCarree(), zorder=10, alpha=1.0)
    if make_hailcast_probabilistic:
        cmap = plt.get_cmap('Blues')
        new_cmap = colors.LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=0.15, b=1.0), cmap(np.linspace(0.15, 1.0, 100)))
        plt.contourf(lons,lats,hailcast,np.arange(0,61,6)/100, cmap=new_cmap, transform=ccrs.PlateCarree(), vmin=0, vmax=0.6, zorder=11, alpha=0.8)
        # plt.colorbar(fraction=0.043, pad=0.02, ticks = np.arange(0,61,6)/100).set_label(label=hailcast_label,size=30) #Changed padding here since other colorbars removed
    else:
        hailcast_and_true[np.nonzero(hailcast_and_true == 0)] = np.nan
        plt.contourf(lons,lats,hailcast, [0, 1], colors='blue', transform=ccrs.PlateCarree(), zorder=11, alpha=1.0)
        plt.contourf(lons,lats,hailcast_and_true, [0, 1], colors='orange', transform=ccrs.PlateCarree(), zorder=12, alpha=1.0)
    pred_val[pred_val < 0.02] = np.nan
    plt.contour(lons,lats,pred_val,np.arange(0,61,6)/100, colors='k', transform=ccrs.PlateCarree(), linewidths=2.3, vmin=0, vmax=0.6, zorder=13)

    if include_reports:
        current_datetime = current_datetime-timedelta(minutes=10)
        loader = StormReportLoader("/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/storm_reports/STORM_EVENTS_2017-2021.csv", "NOAA", current_datetime.strftime("%Y%m%d%H%M"), 20, 0)

        storm_reports_hail_lats, storm_reports_hail_lons = loader.get_hail_reports()
        storm_reports_hail_sig_lats, storm_reports_hail_sig_lons = loader.get_hail_reports('sig_severe')

        plt.scatter(storm_reports_hail_lons, storm_reports_hail_lats, color='k', s=180, marker='o', edgecolors= "white", transform=ccrs.PlateCarree(), zorder=13, label="Severe Hail")
        plt.scatter(storm_reports_hail_sig_lons, storm_reports_hail_sig_lats, color='k', s=700, marker='*', edgecolors= "white", transform=ccrs.PlateCarree(), zorder=14, label="Sig Severe Hail")

    plt.savefig(os.path.join(plot_output_dir, "truth_plot_" + "{:04d}".format(plot_num) + '.png'), bbox_inches='tight')


def load_line_data(path_str):
    if path_str[-1] == "*":
        file_list = glob.glob(path_str)
        file_list.sort()
        ds = xr.open_mfdataset(file_list, concat_dim='n_samples', combine='nested', engine='netcdf4')
    else:
        ds = xr.open_dataset(path_str)
    
    return ds


def plot_line_graphs(args):
    unpack_ouputs = None
    limit_axis = None

    ##### UNUSED single ens member plot #######

    # model_outputs = ["/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/hailcast_refl/examples/*",
    #                 "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/logreg/predictions/y_hats.nc",
    #                 "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/logreg/predictions_ensemble/y_hats.nc",
    #                 "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/all_gaus_runs/predictions/y_hats.nc",
    #                 "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/all_gaus_runs/predictions/y_hats_one_mem.nc"]
    # labels = ["/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/hailcast_refl/labels/*",
    #           "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/logreg/processed/labels/*",
    #           "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/logreg/ensemble_processed/labels/*",
    #           "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/all_gaus_runs/processed/labels/*",
    #           "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/all_gaus_runs/ensemble_processed/labels/*"]
    # line_linestyles = ['-.', '-', '--', '-', '-']
    # line_markers = ['o', 'o', 'o', 'o', 'o']
    # colors = ['k', 'k', 'k', 'b', 'r']
    # line_titles = ["Ens. Hailcast Baseline", "Det. UH Logistic Regression", "Ens. UH Logistic Regression", "Det. Unet w/ Ens. Performance", "Det. Unet w/ Det. Performance"]
    # var_names = [("hailcast", "MESH_class_bin_severe"), ("MESH_class_bin_severe", "MESH_class_bin_severe"), ("MESH_class_bin_severe", "MESH_class_bin_severe"), ("MESH_class_bin", "MESH_class_bin"),
    #               ("MESH_class_bin", "MESH_class_bin")]
    # # main_title = 'Neighborhood Max CSI'
    # main_title = 'Pixelwise Max CSI'
    # output_path = "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/images/test_dataset_plots/c_one_mem.png"
    # xlabel = 'Time Since WoFS Init (mins)'
    # ylabel = 'Pixelwise Max CSI'
    # # ylabel = '6 km Radius Neighborhood Max CSI'
    # show_legend = False
    # plot_mode = "c"
    # include_obs_cuttoff = True

    ##### refl comparison plot #######
    
    # model_outputs = ["/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/no_lightning/predictions/y_hats.nc",
    #                  "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/unprocessed_refl/examples/*",
    #                  "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/no_shuffle/predictions/y_hats.nc",
    #                  "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/unprocessed_no_shuffle/examples/*"]
    # labels = ["/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/no_lightning/processed/labels/*",
    #           "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/unprocessed_refl/examples/*",
    #           "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/no_shuffle/processed/labels/*",
    #           "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/unprocessed_no_shuffle/examples/*"]
    # line_linestyles = ['-', '-', '-', '-']
    # line_markers = ['o', '^', 'o', '^']
    # colors = ['b', 'b', 'r', 'r']
    # line_titles = ["Shuffled ML Model", "Shuffled WoFS Convection", "Unshuffled ML Model", "Unshuffled WoFS Convection"]
    # var_names = [("MESH_class_bin", "MESH_class_bin"), ("comp_dz_class_bin", "ZH_class_bin"), ("MESH_class_bin_severe", "MESH_class_bin_severe"), ("comp_dz_class_bin", "ZH_class_bin")]
    # unpack_ouputs = [False, True, False, True]
    # main_title = 'Model Dependence on WoFS Performance'
    # output_path = "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/images/test_dataset_plots/c_refl_comparison.png"
    # xlabel = 'Time Since WoFS Init (mins)'
    # ylabel = 'Pixelwise Max CSI'
    # # ylabel = '6 km Radius Neighborhood Max CSI'
    # show_legend = True
    # plot_mode = "c"
    # include_obs_cuttoff = False

    ############## Arc ############

    # model_outputs = ["/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/hailcast_refl/examples/*",
    #                 "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/logreg/predictions/y_hats.nc",
    #                 "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/logreg/predictions_ensemble/y_hats.nc",
    #                 "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/all_gaus_runs/predictions/y_hats.nc",
    #                 "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/all_gaus_runs/predictions_ensemble/y_hats.nc",
    #                 "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/multi_models/2d_preds/y_hats.nc",
    #                 "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/multi_models/2d_preds/y_hats_ens.nc",
    #                 "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/single_model/single_preds/y_hats.nc",
    #                 "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/single_model/single_preds/y_hats_ens.nc"]
    # labels = ["/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/hailcast_refl/labels/*",
    #           "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/logreg/processed/labels/*",
    #           "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/logreg/ensemble_processed/labels/*",
    #           "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/all_gaus_runs/processed/labels/*",
    #           "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/all_gaus_runs/ensemble_processed/labels/*",
    #           "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/single_model/processed/labels/*",
    #           "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/single_model/ensemble_processed/labels/*",
    #           "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/single_model/processed/labels/*",
    #           "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/single_model/ensemble_processed/labels/*"]
    # line_linestyles = ['-', '-', '-', '-', '-', '-', '-', '-', '-']
    # line_markers = ['s', 'o', '^', 'o', '^', 'o', '^', 'o', '^']
    # colors = ['k', 'k', 'k', 'b', 'b', 'r', 'r', 'y', 'y']
    # line_titles = ["Ens. Hailcast Baseline", "Det. UH Logistic Regression", "Ens. UH Logistic Regression", "Det. Architecture Exp. 3", "Ens. Architecture Exp. 3", 
    #                "Det. Architecture Exp. 2", "Ens. Architecture Exp. 2", "Det. Architecture Exp. 1", "Ens. Architecture Exp. 1"]
    # var_names = [("hailcast", "MESH_class_bin_severe"), ("MESH_class_bin_severe", "MESH_class_bin_severe"), ("MESH_class_bin_severe", "MESH_class_bin_severe"), ("MESH_class_bin", "MESH_class_bin"),
    #              ("MESH_class_bin", "MESH_class_bin"), ("MESH_class_bin_severe_0", "MESH_class_bin_severe"), ("MESH_class_bin_severe_0", "MESH_class_bin_severe"), ("MESH_class_bin_severe", "MESH_class_bin_severe"),
    #                ("MESH_class_bin_severe", "MESH_class_bin_severe")]
    # # main_title = 'Neighborhood Max CSI'
    # main_title = 'Pixelwise Max CSI'
    # output_path = "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/images/test_dataset_plots/c_arcs.png"
    # xlabel = 'Time Since Forecast Start (mins)'
    # ylabel = 'Pixelwise Max CSI'
    # # ylabel = '6 km Radius Neighborhood Max CSI'
    # show_legend = False
    # plot_mode = "c"
    # include_obs_cuttoff = True

    ######### ROC and Re curves #########

    # model_outputs = ["/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/all_gaus_runs/predictions/y_hats.nc",
    #                  "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/all_gaus_runs/predictions/y_hats.nc",
    #                  "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/all_gaus_runs/predictions/y_hats.nc",
    #                  "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/all_gaus_runs/predictions/y_hats.nc",
    #                  "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/all_gaus_runs/predictions/y_hats.nc"]
    # labels = ["/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/all_gaus_runs/processed/labels/*",
    #           "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/all_gaus_runs/processed/labels/*",
    #           "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/all_gaus_runs/processed/labels/*",
    #           "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/all_gaus_runs/processed/labels/*",
    #           "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/all_gaus_runs/processed/labels/*"]
    # line_linestyles = ['-', '-', '-', '-', '-']
    # line_markers = ['', '', '', '', '']
    # colors = ['b', 'g', 'r', 'c', 'y']
    # line_titles = ["ML Start + 00 mins", "ML Start + 10 mins", "ML Start + 20 mins", "ML Start + 30 mins", "ML Start + 40 mins"]
    # var_names = [("MESH_class_bin", "MESH_class_bin"), ("MESH_class_bin", "MESH_class_bin"), ("MESH_class_bin", "MESH_class_bin"), ("MESH_class_bin", "MESH_class_bin"),
    #               ("MESH_class_bin", "MESH_class_bin")]
    # main_title = 'Pixelwise Reliability Diagram'
    # output_path = "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/images/test_dataset_plots/Reliability_Diagram.png"
    # xlabel = 'Predicted Probability'
    # ylabel = 'Observed Frequency'
    # # main_title = 'Pixelwise Receiver Operating Characteristic Curve'
    # # output_path = "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/images/test_dataset_plots/ROC.png"
    # # xlabel = 'False Positive Rate'
    # # ylabel = 'True Positive Rate'
    # show_legend = True
    # plot_mode = "re"
    # include_obs_cuttoff = False

    ######### Gaus #########

    # model_outputs = ["/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/hailcast_refl/examples/*",
    #                  "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/logreg/predictions/y_hats.nc",
    #                  "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/logreg/predictions_ensemble/y_hats.nc",
    #                  "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/all_gaus_runs/predictions/y_hats.nc",
    #                  "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/all_gaus_runs/predictions/y_hats_nogaus.nc",
    #                  "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/all_gaus_runs/predictions_ensemble/y_hats.nc",
    #                  "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/all_gaus_runs/predictions_ensemble/y_hats_nogaus.nc"]
    # labels = ["/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/hailcast_refl/labels/*",
    #           "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/logreg/processed/labels/*",
    #           "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/logreg/ensemble_processed/labels/*",
    #           "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/all_gaus_runs/processed/labels/*",
    #           "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/all_gaus_runs/processed/labels/*",
    #           "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/all_gaus_runs/ensemble_processed/labels/*",
    #           "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/all_gaus_runs/ensemble_processed/labels/*"]
    # line_linestyles = ['-', '-', '-', '-', '-', '-', '-']
    # line_markers = ['s', 'o', '^', 'o', 'o', '^', '^']
    # colors = ['k', 'k', 'k', 'b', 'r', 'b', 'r']
    # line_titles = ["Ens. Hailcast Baseline", "Det. UH Logistic Regression", "Ens. UH Logistic Regression", "Det. Unet with Expansion", "Det. Unet w/o Expansion",
    #                "Ens. Unet with Expansion", "Ens. Unet w/o Expansion"]
    # var_names = [("hailcast", "MESH_class_bin_severe"), ("MESH_class_bin_severe", "MESH_class_bin_severe"), ("MESH_class_bin_severe", "MESH_class_bin_severe"), ("MESH_class_bin", "MESH_class_bin"),
    #               ("MESH_class_bin", "MESH_class_bin"), ("MESH_class_bin", "MESH_class_bin"), ("MESH_class_bin", "MESH_class_bin")]
    # # main_title = 'Neighborhood Max CSI'
    # main_title = 'Pixelwise Max CSI'
    # output_path = "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/images/test_dataset_plots/c_gaus.png"
    # xlabel = 'Time Since Forecast Start (mins)'
    # ylabel = 'Pixelwise Max CSI'
    # # ylabel = '6 km Radius Neighborhood Max CSI'
    # show_legend = False
    # plot_mode = "c"
    # include_obs_cuttoff = True

    ######### Lightning/main CSI plots #########

    model_outputs = ["/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/hailcast_refl/examples/*",
                    "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/logreg/predictions/y_hats.nc",
                    "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/logreg/predictions_ensemble/y_hats.nc",
                    "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/all_gaus_runs/predictions/y_hats.nc",
                    "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/no_lightning/predictions/y_hats.nc",
                    # "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/all_gaus_runs/predictions_ensemble/y_hats.nc",
                    "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/no_lightning/predictions_ensemble/y_hats.nc"]
    labels = ["/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/hailcast_refl/labels/*",
              "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/logreg/processed/labels/*",
              "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/logreg/ensemble_processed/labels/*",
              "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/all_gaus_runs/processed/labels/*",
              "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/no_lightning/processed/labels/*",
              # "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/all_gaus_runs/ensemble_processed/labels/*",
              "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/no_lightning/ensemble_processed/labels/*"]
    # line_linestyles = ['-', '-', '-', '-', '-', '-', '-']
    # line_markers = ['s', 'o', '^', 'o', 'o', '^', '^']
    # colors = ['k', 'k', 'k', 'b', 'r', 'b', 'r']
    # line_linestyles = ['-', '-', '-', '-', '-', '-']
    line_markers = ['s', 'o', '^', 'o', 'o', '^']
    # colors = ['k', 'k', 'k', 'b', 'r', 'r']
    line_linestyles = ['-', '-', '-', '-', '-', '-']
    # line_markers = ['o', 'o', 'o', 'o', 'o', 'o']
    colors = ['k', 'k', 'k', 'b', 'r', 'r']
    # line_titles = ["Ens. Hailcast Baseline", "Det. UH Logistic Regression", "Ens. UH Logistic Regression", "Det. Unet with Lightning", "Det. Unet w/o Lightning",
    #                "Ens. Unet with Lightning", "Ens. Unet w/o Lightning"]
    # var_names = [("hailcast", "MESH_class_bin_severe"), ("MESH_class_bin_severe", "MESH_class_bin_severe"), ("MESH_class_bin_severe", "MESH_class_bin_severe"), ("MESH_class_bin", "MESH_class_bin"),
    #               ("MESH_class_bin", "MESH_class_bin"), ("MESH_class_bin", "MESH_class_bin"), ("MESH_class_bin", "MESH_class_bin")]
    line_titles = ["Ens. Hailcast Baseline", "Det. UH Logistic Regression", "Ens. UH Logistic Regression", "Unet with Lightning", "Det. Unet w/o Lightning",
                   "Ens. Unet w/o Lightning"]
    var_names = [("hailcast", "MESH_class_bin_severe"), ("MESH_class_bin_severe", "MESH_class_bin_severe"), ("MESH_class_bin_severe", "MESH_class_bin_severe"), ("MESH_class_bin", "MESH_class_bin"),
                  ("MESH_class_bin", "MESH_class_bin"), ("MESH_class_bin", "MESH_class_bin")]
    main_title = 'Neighborhood Max CSI'
    # main_title = 'Pixelwise Max CSI'
    # output_path = "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/images/test_dataset_plots/c_new_main_plot.png"
    output_path = "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/images/test_dataset_plots/cn_new_main_plot.png"
    limit_axis = True
    xlabel = 'Time Since ML Forecast Start (mins)'
    # ylabel = 'Pixelwise Max CSI'
    ylabel = '6 km Radius Neighborhood Max CSI'
    show_legend = True
    plot_mode = "cn"
    include_obs_cuttoff = False

    ######### Small hail #########

    # model_outputs = ["/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/hailcast_refl/examples/*",
    #                 "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/logreg/predictions/y_hats.nc",
    #                 "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/logreg/predictions_ensemble/y_hats.nc",
    #                 "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/all_gaus_runs/predictions/y_hats.nc",
    #                 "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/small_hail/predictions/y_hats.nc",
    #                 "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/all_gaus_runs/predictions_ensemble/y_hats.nc",
    #                 "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/small_hail/predictions_ensemble/y_hats.nc"]
    # labels = ["/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/hailcast_refl/labels/*",
    #           "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/logreg/processed/labels/*",
    #           "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/logreg/ensemble_processed/labels/*",
    #           "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/all_gaus_runs/processed/labels/*",
    #           "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/small_hail/processed/labels/*",
    #           "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/all_gaus_runs/ensemble_processed/labels/*",
    #           "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/small_hail/ensemble_processed/labels/*"]
    # line_linestyles = ['-', '-', '-', '-', '-', '-', '-']
    # line_markers = ['s', 'o', '^', 'o', 'o', '^', '^']
    # colors = ['k', 'k', 'k', 'b', 'r', 'b', 'r']
    # line_titles = ["Ens. Hailcast Baseline", "Det. UH Logistic Regression", "Ens. UH Logistic Regression", "Severe Hail Det. Unet", "Any Hail Det. Unet",
    #                "Severe Hail Ens. Unet", "Any Hail Ens. Unet"]
    # var_names = [("hailcast", "MESH_class_bin_severe"), ("MESH_class_bin_severe", "MESH_class_bin_severe"), ("MESH_class_bin_severe", "MESH_class_bin_severe"), ("MESH_class_bin", "MESH_class_bin"),
    #               ("MESH_class_bin", "MESH_class_bin"), ("MESH_class_bin", "MESH_class_bin"), ("MESH_class_bin", "MESH_class_bin")]
    # main_title = 'Neighborhood Max CSI'
    # # main_title = 'Pixelwise Max CSI'
    # output_path = "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/images/test_dataset_plots/cn_smallhail.png"
    # xlabel = 'Time Since Forecast Start (mins)'
    # # ylabel = 'Pixelwise Max CSI'
    # ylabel = '6 km Radius Neighborhood Max CSI'
    # show_legend = True
    # plot_mode = "cn"
    # include_obs_cuttoff = True

     ######### Performance Diags #########

    # model_outputs = ["/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/hailcast_refl/examples/*",
    #                 "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/logreg/predictions/y_hats.nc",
    #                 "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/logreg/predictions_ensemble/y_hats.nc",
    #                 "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/all_gaus_runs/predictions/y_hats.nc",
    #                 "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/no_lightning/predictions/y_hats.nc",
    #                 # "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/all_gaus_runs/predictions_ensemble/y_hats.nc",
    #                 "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/no_lightning/predictions_ensemble/y_hats.nc"]
    # labels = ["/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/hailcast_refl/labels/*",
    #           "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/logreg/processed/labels/*",
    #           "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/logreg/ensemble_processed/labels/*",
    #           "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/all_gaus_runs/processed/labels/*",
    #           "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/no_lightning/processed/labels/*",
    #           # "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/all_gaus_runs/ensemble_processed/labels/*",
    #           "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/test/no_lightning/ensemble_processed/labels/*"]
    # line_markers = ['', '', '', '', '', '']
    # line_linestyles = ['-', (0, (5, 10)), '-', (0, (5, 10)), (0, (5, 10)), '-']
    # colors = ['k', 'y', 'y', 'b', 'r', 'r']
    # line_titles = ["Ens. Hailcast Baseline", "Det. UH Logistic Regression", "Ens. UH Logistic Regression", "Unet with Lightning", "Det. Unet w/o Lightning",
    #                "Ens. Unet w/o Lightning"]
    # var_names = [("hailcast", "MESH_class_bin_severe"), ("MESH_class_bin_severe", "MESH_class_bin_severe"), ("MESH_class_bin_severe", "MESH_class_bin_severe"), ("MESH_class_bin", "MESH_class_bin"),
    #               ("MESH_class_bin", "MESH_class_bin"), ("MESH_class_bin", "MESH_class_bin")]
    # main_title = 'Pixelwise Performance Diagram at t=40 Minutes'
    # output_path = "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/images/test_dataset_plots/perf1.png"
    # show_legend = True
    # plot_mode = "perf"
    # include_obs_cuttoff = False
    # perf_time_index = 11

    ############################
    if unpack_ouputs is None:
        unpack_ouputs = np.zeros(len(model_outputs))

    lead_time_minutes = np.arange(0,60,5)
    lead_time_minutes = lead_time_minutes - 15
    lead_time_indices = [3, 5, 7, 9, 11]

    n_boot = 200
    boot_size = 400

    ################## TEMP ################
    if plot_mode == "c":
        # all_plot_data = [all_plot_data[2], all_plot_data[3], all_plot_data[5]]
        line_linestyles = [line_linestyles[2], line_linestyles[3], line_linestyles[5]]
        line_markers = [line_markers[2], line_markers[3], line_markers[5]]
        line_titles = [line_titles[2], line_titles[3], line_titles[5]]
        colors = [colors[2], colors[3], colors[5]]
        model_outputs = [model_outputs[2], model_outputs[3], model_outputs[5]]
        labels = [labels[2], labels[3], labels[5]]
        var_names = [var_names[2], var_names[3], var_names[5]]
        unpack_ouputs = [unpack_ouputs[2], unpack_ouputs[3], unpack_ouputs[5]]
    if plot_mode == "c" or plot_mode == "cn":
        lead_time_minutes = lead_time_minutes[3:]
    ########################################

    all_plot_data = []
    confidence_polygons = []

    for j, (model_output, label, var_name, unpack_ouput, color) in enumerate(zip(model_outputs, labels, var_names, unpack_ouputs, colors)):
        print("Reached:")
        print(var_name)

        pred_ds = load_line_data(model_output)
        truth_ds = load_line_data(label)

        pred_ds = pred_ds.transpose(..., "time_dim")
        truth_ds = truth_ds.transpose(..., "time_dim")

        if unpack_ouput:
            truth_ds = truth_ds[var_name[1]]

            truth_ds = truth_ds.transpose("n_samples", "lat_dim", "lon_dim", "time_dim")

            current_label_data = unpack_ne_dim_output(truth_ds.to_numpy(), 18, 1)
        
        else:
            current_label_data = truth_ds[var_name[1]].to_numpy()

        current_pred_data = pred_ds[var_name[0]].to_numpy()

        plot_data = []

        ################## TEMP ################
        if plot_mode == "c" or plot_mode == "cn":
            current_label_data = current_label_data[...,3:]
            if current_pred_data.shape[-1] != 9:
                current_pred_data = current_pred_data[...,3:]
        ########################################

        # if current_pred_data.shape[-1] == 9:
        #     plot_data = [np.nan, np.nan, np.nan]
        #     current_label_data = current_label_data[...,3:]

        if plot_mode == "c":
            for i in range(current_pred_data.shape[-1]):
                single_pred_data = current_pred_data[...,i]
                single_label_data = current_label_data[...,i]
                single_pred_data = single_pred_data.flatten()
                single_label_data = single_label_data.flatten()

                max_csi_metric = MaxCriticalSuccessIndex()
                max_csi_metric.reset_state()
                max_csi_metric.update_state(single_label_data, single_pred_data)
                plot_data.append(max_csi_metric.result().numpy())
            
            confidence_polygons.append(make_confidence_intervals(lead_time_minutes, current_pred_data, current_label_data, color, n_boot, boot_size, nbhr=False))

        elif plot_mode == "cn":
            for i in range(current_pred_data.shape[-1]):
                single_pred_data = current_pred_data[...,i]
                single_label_data = current_label_data[...,i]
                single_pred_data = np.expand_dims(single_pred_data, -1)
                single_label_data = np.expand_dims(single_label_data, -1)

                plot_data.append(neighborhood_max_csi(single_label_data, single_pred_data))
            
            confidence_polygons.append(make_confidence_intervals(lead_time_minutes, current_pred_data, current_label_data, color, n_boot, boot_size, nbhr=True))

        elif plot_mode == "re":
            single_pred_data = current_pred_data[...,lead_time_indices[j]]
            single_label_data = current_label_data[...,lead_time_indices[j]]
            single_pred_data = single_pred_data.flatten()
            single_label_data = single_label_data.flatten()

            prob_true, prob_pred = calibration_curve(single_label_data, single_pred_data, n_bins=100)

            plot_data = (prob_pred, prob_true)
        
        elif plot_mode == "roc":
            single_pred_data = current_pred_data[...,lead_time_indices[j]]
            single_label_data = current_label_data[...,lead_time_indices[j]]
            single_pred_data = single_pred_data.flatten()
            single_label_data = single_label_data.flatten()

            auc_score = roc_auc_score(single_label_data, single_pred_data)
            fpr, tpr, thresholds = roc_curve(single_label_data, single_pred_data)
            plot_data = (fpr, tpr, auc_score)
            
        elif plot_mode == "perf":
            single_pred_data = current_pred_data[..., perf_time_index]
            single_label_data = current_label_data[..., perf_time_index]
            single_pred_data = single_pred_data.flatten()
            single_label_data = single_label_data.flatten()

            #compute the score
            precision, recall, thresh, csi = get_performance_nums(single_label_data, single_pred_data)
            plot_data = (precision, recall, thresh, csi)

        else:
            raise Exception("Please give valid plot_mode")

        all_plot_data.append(plot_data)

    ################## TEMP ################
    if plot_mode == "cn":
        confidence_polygons = [confidence_polygons[2],confidence_polygons[3],confidence_polygons[5]]
    ########################################

    # Make the actual plots
    if plot_mode == "perf":
        fig = plt.figure(figsize=(20, 20), dpi=150)
    else:
        fig = plt.figure(figsize=(16, 12), dpi=300)
    fig.patch.set_facecolor('white')
    if limit_axis is not None and limit_axis:
        plt.xlim([-0.5, 40.5])
        plt.ylim([0, 0.75])
    for plot_data, line_linestyle, line_marker, line_title, color in zip(all_plot_data, line_linestyles, line_markers, line_titles, colors):
        if plot_mode == "re":
            plt.plot(plot_data[0], plot_data[1], linestyle=line_linestyle, label=line_title, marker=line_marker, linewidth=2, markersize=12, c=color)
        elif plot_mode == "roc":
            plt.plot(plot_data[0], plot_data[1], linestyle=line_linestyle, label=line_title + ': AUC=%.3f' % (plot_data[2]), marker=line_marker, linewidth=2, markersize=12, c=color)
        elif plot_mode == "perf":
            plt.plot(plot_data[0], plot_data[1], linestyle=line_linestyle, label=line_title, marker=line_marker, linewidth=2, markersize=12, c=color)
            max_csi = np.argmax(plot_data[3])
            plt.scatter(plot_data[0][max_csi], plot_data[1][max_csi], s=400, marker='x', color='black', linewidth = 5)
        else:
            plt.plot(lead_time_minutes, plot_data, linestyle=line_linestyle, label=line_title, marker=line_marker, linewidth=2, markersize=12, c=color)
    
    if include_obs_cuttoff:
        plt.axvline(x = 0, linewidth=2, color = 'g', label = 'Observations Threshold')
    if plot_mode == "re":
        plt.plot([0,1], linestyle='--', c='k')
    if plot_mode == "roc":
        plt.plot([0,1], [0,1], linestyle='--', c='k', label='No Skill: AUC=%.3f' % (0.5))
    if plot_mode == "c" or plot_mode == "cn":
        ax = fig.axes[0]
        for confidence_polygon in confidence_polygons:
            ax.add_patch(confidence_polygon)
    if plot_mode == "perf":
        #apply background and reference lines
        ax = fig.axes[0]
        ax, cf4 = make_performance(ax, main_title)
    else:
        plt.ylabel(ylabel, size=30)
        plt.xlabel(xlabel, size=30)
        plt.yticks(fontsize=28)
        plt.xticks(fontsize=28)
        plt.title(main_title, size=40)
        if show_legend:
            plt.legend(prop={'size': 20})
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')


def get_performance_nums(truth,pred):
    #make performance curve
    precision, recall, thresh = precision_recall_curve(truth, pred)

    #get CSI
    csi = 1 / ((1/precision) + (1/recall) - 1)

    return precision, recall, thresh, csi


def make_performance(ax, title):

    """
    Make a performance diagram (Roebber 2009). 
    """
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    xx = np.linspace(0.001,1,100)
    yy = xx
    xx,yy = np.meshgrid(xx,xx)
    csi = 1 / (1/xx + 1/yy -1)
    cf = ax.contourf(xx,yy,csi, cmap='Blues', alpha=0.3, levels=np.arange(0,1.1,0.1))
    ax.set_xlabel('Success Ratio (SR; 1-FAR)', fontsize = 30)
    ax.set_ylabel('Probability of Detection (POD)', fontsize = 30)
    biasLines = ax.contour(
                xx,
                yy,
                yy/xx,
                colors="k",
                levels=[0.5, 1.0, 1.5, 2.0, 4.0],
                linestyles="dashed",
                linewidths=1,
                alpha=0.9
                )
    ax.clabel(biasLines, levels=[0.5, 1.0, 1.5, 2.0, 4.0], fontsize=28, inline=True, fmt="%1.1f")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig = ax.get_figure()
    cbar = fig.colorbar(cf, cax=cax)
    cbar.ax.tick_params(labelsize=28)
    ax.tick_params(axis='both', which='major', labelsize=28)
    cbar.set_label(label='Critical Success Index (CSI)', size=30)
    fig.legend(fontsize=28, bbox_to_anchor=(0.9, 0.95))
    fig.suptitle(title, size = 40)

    return ax, cf


if __name__ == "__main__":
    # Perform all the argument parsing
    parser = create_parser()
    args = parser.parse_args()
    args = vars(args)

    plot_animation = args["plot_animation"]
    save_stitched_domain = args["save_stitched_domain"]
    plot_stitched_domain = args["plot_stitched_domain"]

    if plot_animation or save_stitched_domain:
        plot_casestudy_plots(args)
    elif plot_stitched_domain:
        plot_averaged_casestudy_plots(args)
    else:
        plot_line_graphs(args)