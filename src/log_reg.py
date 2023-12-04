import xarray as xr
import argparse
import pickle
import glob
import numpy as np
from sklearn.linear_model import LogisticRegression
from custom_metrics import MaxCriticalSuccessIndex
import random


EXAMPLES_VAR_NAME = "uh_2to5"
LABELS_VAR_NAME = "MESH_class_bin_severe"


def create_parser():
    '''
    Create argument parser
    '''
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description='Logistic Regression', fromfile_prefix_chars='@')

    parser.add_argument('--examples', type=str, default='/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour-more_fields-1_inch-cross_val/patches/animations/20190501/unprocessed/examples/*')
    parser.add_argument('--labels', type=str, default='/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour-more_fields-1_inch-cross_val/patches/animations/20190501/unprocessed/examples/*')
    parser.add_argument('--val_examples', type=str, default='/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour-more_fields-1_inch-cross_val/patches/animations/20190501/unprocessed/examples/*')
    parser.add_argument('--val_labels', type=str, default='/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour-more_fields-1_inch-cross_val/patches/animations/20190501/unprocessed/examples/*')
    parser.add_argument('--saved_model_path', type=str, default='/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour-more_fields-1_inch-cross_val/patches/animations/20190501/unprocessed/examples/*')
    parser.add_argument('--test_examples', type=str, default='/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour-more_fields-1_inch-cross_val/patches/animations/20190501/unprocessed/examples/*')
    parser.add_argument('--predictions_outfile', type=str, default='/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-1_hour-more_fields-1_inch-cross_val/patches/animations/20190501/unprocessed/examples/*')

    parser.add_argument('--predict', '-p', action='store_true')
    parser.add_argument('--n_cpus', type=int, default=12)

    return parser


# NOTE: Assumes data does not have "ne" dim right now. 3D data should be fine.
def train(examples, labels, val_examples, val_labels, saved_model_path, n_cpus):
    # Hyperparams
    solvers = ["sag", "saga"] #["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"]
    reg_strengths = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]

    examples_list = glob.glob(examples)
    examples_list.sort()
    examples_ds = xr.open_mfdataset(examples_list, concat_dim='n_samples', combine='nested', engine='netcdf4')
    examples_ds = examples_ds[EXAMPLES_VAR_NAME]

    examples_list = glob.glob(val_examples)
    examples_list.sort()
    examples_val_ds = xr.open_mfdataset(examples_list, concat_dim='n_samples', combine='nested', engine='netcdf4')
    examples_val_ds = examples_val_ds[EXAMPLES_VAR_NAME]

    labels_list = glob.glob(labels)
    labels_list.sort()
    labels_ds = xr.open_mfdataset(labels_list, concat_dim='n_samples', combine='nested', engine='netcdf4')
    labels_ds = labels_ds[LABELS_VAR_NAME]

    labels_list = glob.glob(val_labels)
    labels_list.sort()
    labels_val_ds = xr.open_mfdataset(labels_list, concat_dim='n_samples', combine='nested', engine='netcdf4')
    labels_val_ds = labels_val_ds[LABELS_VAR_NAME]

    # Make problem pixelwise. Again "ne" must not be present.
    examples_ds = examples_ds.to_numpy().flatten()
    labels_ds = labels_ds.to_numpy().flatten()
    examples_val_ds = examples_val_ds.to_numpy().flatten()
    labels_val_ds = labels_val_ds.to_numpy().flatten()

    # The next two blocks reduce the dataset to 1/2 hail pixels and 1/2 non-hail. Otherwise it is way too massive.
    examples_ds_hail = examples_ds[np.nonzero(labels_ds == 1)]
    labels_ds_hail = labels_ds[np.nonzero(labels_ds == 1)]
    examples_ds_empty = examples_ds[np.nonzero(labels_ds == 0)]
    labels_ds_empty = labels_ds[np.nonzero(labels_ds == 0)]
    chosen_empty_pixels = np.random.choice(len(examples_ds_empty), len(examples_ds_hail), replace=False)
    examples_ds_empty = examples_ds_empty[chosen_empty_pixels]
    labels_ds_empty = labels_ds_empty[chosen_empty_pixels]

    examples_ds = np.concatenate((examples_ds_hail, examples_ds_empty))
    labels_ds = np.concatenate((labels_ds_hail, labels_ds_empty))
    combined_lists_for_shuffle = list(zip(examples_ds, labels_ds))
    random.shuffle(combined_lists_for_shuffle)
    examples_ds, labels_ds = zip(*combined_lists_for_shuffle)
    examples_ds = np.array(examples_ds)
    labels_ds = np.array(labels_ds)

    examples_ds = np.expand_dims(examples_ds, axis=-1)
    examples_val_ds = np.expand_dims(examples_val_ds, axis=-1)

    # Manually perform hparam grid search
    best_csi = 0
    best_model = None
    for sovler in solvers:
        for reg_strength in reg_strengths:
            model = LogisticRegression(multi_class="ovr", max_iter=1000, solver=sovler, C=1/reg_strength, n_jobs=n_cpus, verbose=1)
            model.fit(examples_ds, labels_ds)

            y_hat = model.predict(examples_val_ds)

            max_csi_metric = MaxCriticalSuccessIndex()
            max_csi_metric.reset_state()
            max_csi_metric.update_state(labels_val_ds, y_hat)
            max_csi_metric_result = max_csi_metric.result().numpy()

            print("Max CSI: " + str(max_csi_metric_result))

            if max_csi_metric_result >= best_csi:
                best_csi = max_csi_metric_result
                best_model = model

    pickle.dump(best_model, open(saved_model_path, 'wb'))


# NOTE: examples arg here is for the test set.
# This time the dataset must be 3D. Assumes "time_dim" present
def predict(examples, saved_model_path, predictions_outfile):
    examples_list = glob.glob(examples)
    examples_list.sort()
    examples_ds = xr.open_mfdataset(examples_list, concat_dim='n_samples', combine='nested', engine='netcdf4')

    model = pickle.load(open(saved_model_path, 'rb'))

    data_for_all_timesteps = []
    for i in range(examples_ds.dims["time_dim"]):
        if "ne" in examples_ds.dims:
            data_for_all_ensemble_members = []

            for j in range(examples_ds.dims["ne"]):
                example_ds_one_time = examples_ds[{"time_dim": i}]
                example_ds_one_time = example_ds_one_time[{"ne": j}]
                example_ds_one_time = example_ds_one_time[EXAMPLES_VAR_NAME].to_numpy().flatten()
                example_ds_one_time = np.expand_dims(example_ds_one_time, axis=-1)

                data_for_all_ensemble_members.append(model.predict(example_ds_one_time))

            all_ensemble_data_stacked = np.stack(data_for_all_ensemble_members, axis=-1)
            data_for_all_timesteps.append(np.mean(all_ensemble_data_stacked, axis=-1))
        else:
            example_ds_one_time = examples_ds[{"time_dim": i}]
            
            example_ds_one_time = example_ds_one_time[EXAMPLES_VAR_NAME].to_numpy().flatten()
            example_ds_one_time = np.expand_dims(example_ds_one_time, axis=-1)

            data_for_all_timesteps.append(model.predict(example_ds_one_time))

    output_np = np.stack(data_for_all_timesteps, axis=-1)
    new_ds_dict = {LABELS_VAR_NAME:(("n_samples", "time_dim"), output_np)}
    output_ds = xr.Dataset(data_vars = new_ds_dict)

    output_ds.to_netcdf(predictions_outfile)


if __name__ == "__main__":
    # Perform all the argument parsing
    parser = create_parser()
    args = parser.parse_args()
    args = vars(args)

    if args["predict"]:
        predict(args["test_examples"], args["saved_model_path"], args["predictions_outfile"])
    else:
        train(args["examples"], args["labels"], args["val_examples"], args["val_labels"], args["saved_model_path"], args["n_cpus"])