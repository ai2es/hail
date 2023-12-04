# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Write sample summary data for the hparams plugin.
See also `hparams_minimal_demo.py` in this directory for a demo that
runs much faster, using synthetic data instead of actually training
MNIST models.
"""

import tensorflow as tf
import os

# This block grabs my GPUs
if "CUDA_VISIBLE_DEVICES" in os.environ.keys():
    # Fetch list of logical GPUs that have been allocated
    #  Will always be numbered 0, 1, …
    physical_devices = tf.config.get_visible_devices('GPU')
    n_physical_devices = len(physical_devices)

    # Set memory growth for each
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
else:
	# No allocated GPUs: do not delete this case!                                                                	 
    tf.config.set_visible_devices([], 'GPU')

tf.config.threading.set_intra_op_parallelism_threads(16)
tf.config.threading.set_inter_op_parallelism_threads(16)


import random
import shutil
from absl import app
from absl import flags
import numpy as np
from tensorboard.plugins.hparams import api as hp
from keras_unet_collection import models
from operator import itemgetter
from custom_metrics import MaxCriticalSuccessIndex
from custom_metrics import WeightedBinaryCrossEntropy
from custom_metrics import fractions_skill_score_loss
from custom_metrics import EarlyStoppingLoss
import glob
import warnings

if int(tf.__version__.split(".")[0]) < 2:
    # The tag names emitted for Keras metrics changed from "acc" (in 1.x)
    # to "accuracy" (in 2.x), so this demo does not work properly in
    # TensorFlow 1.x (even with `tf.enable_eager_execution()`).
    raise ImportError("TensorFlow 2.x is required to run this demo.")


flags.DEFINE_integer(
    "num_session_groups",
    1500, # Was 300
    "The approximate number of session groups to create.",
)
flags.DEFINE_string(
    "logdir",
    "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/saved_models_refl_nolightning/fold_0000/tensorboard_logdir",
    "The directory to write the summary information to.",
)
flags.DEFINE_integer(
    "summary_freq",
    600,
    "Summaries will be written every n steps, where n is the value of "
    "this flag.",
)
flags.DEFINE_integer(
    "num_epochs",
    80, # was 150 for multi-2d-case!
    "Number of epochs per trial.",
)

# my params
TF_TRAIN_DS_PATH_GLOB = "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/cv_folds_gridrad_refl/fold_0000/train/tf_datasets_no_lightning/*"
TF_VAL_DS_PATH_GLOB = "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/patches/cv_folds_gridrad_refl/fold_0000/val/tf_datasets_no_lightning/*"
H5_MODELS_DIR = "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/saved_models_refl_nolightning/fold_0000/h5_models"
CHECKPOINTS_DIR = "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/3d_unets-2d_unets-FINAL/saved_models_refl_nolightning/fold_0000/checkpoints"
# OLD TESTS: Was 71370 for fold 0004 72144 for fold 0003 72036 for fold 0002 72414 for fold 0001 and 71316 for fold 0000
# WITH ALL OPTIONS MODELS: Was 60732 for fold 0004 60660 for fold 0003 60678 for fold 0002 60714 for fold 0001 60624 for fold 0000
# For small hail: 60210
# For 2D model 0: 60318 for model 1: 60210 for model 2: 60210 for model 3: 59958 for model 4: 60210 for model 5: 60228 for model 6: 60336
# For model 7: 60174 for model 8: 60192
NUM_SAMPLES_IN_MEM = 60210
MEM_SAMPLES_NUM_IS_COMPLETE_DS_SIZE = True
# Was (64,64,12,19) for 3D case with lightning
# For model 0: (64,64,76) for model 1: (64,64,94) for model 2: (64,64,112) for model 3: (64,64,130) for model 4: (64,64,148) for model 5: (64,64,166)
# For model 6: (64,64,184 for model 7: (64,64,202) for model 8: (64,64,220)
INPUT_SHAPE = (64,64,12,18) #19 for lightning
OUTPUT_CLASSES = 1
OUTPUT_ACTIVATION = "Sigmoid"
VALIDATION_FREQ = 1
# Was 4 for 3D case and 3 for 2D case
PATIENCE = 4
IS_3D_DATA = True
USE_MULTIPLE_GPUS = False
RANDOM_SEED = 3489 # Was 5690
RUN_START_INDEX = [46]
RUN_SINGLE_MODEL = True

#convolution params
HP_CONV_LAYERS = hp.HParam("conv_layers", hp.IntInterval(1, 3))
HP_CONV_KERNEL_SIZE = hp.HParam("conv_kernel_size", hp.Discrete([3, 5, 7]))
HP_CONV_ACTIVATION = hp.HParam("conv_activation", hp.Discrete(['ReLU', 'ELU']))
HP_CONV_KERNELS = hp.HParam('num_of_kernels', hp.Discrete([4,8,16,32]))
HP_LOSS_WEIGHT = hp.HParam('loss_weights', hp.Discrete([2.0,3.0,4.0,5.0,7.0]))
# HP_FSS_RADII = hp.HParam('FSS_radii', hp.Discrete([2,3,4]))

#unet param
HP_UNET_DEPTH = hp.HParam('depth_of_unet', hp.Discrete([1,2,3])) # Was [1,2,3,4,5] for or multi-2d-case!
HP_OPTIMIZER = hp.HParam("optimizer", hp.Discrete(["adam", "adagrad", "sgd", "rmsprop"]))
HP_LOSS = hp.HParam("loss", hp.Discrete(["binary_crossentropy", "weighted_binary_crossentropy"]))
HP_BATCHNORM = hp.HParam('batchnorm', hp.Discrete([False, True]))
HP_3PLUS = hp.HParam('3plus', hp.Discrete([False, True]))
HP_BATCHSIZE = hp.HParam('batch_size', hp.Discrete([32,64,128,256])) # Also had 512
HP_VAL_BATCHSIZE = hp.HParam('val_batch_size', hp.Discrete([128])) # Was 512
HP_LEARNING_RATE = hp.HParam('learning_rate', hp.Discrete([1e-2,1e-3]))
HP_L2_REG = hp.HParam('l2_reg', hp.Discrete([0.1, 0.05, 0.01, 0.005, 0.001, 0.0001, 0.00001]))
HP_L1_REG = hp.HParam('l1_reg', hp.Discrete([0.1, 0.05, 0.01, 0.005, 0.001, 0.0001, 0.00001]))

HPARAMS = [HP_CONV_LAYERS,
    HP_CONV_KERNEL_SIZE,
    HP_CONV_ACTIVATION,
    HP_CONV_KERNELS,
    HP_UNET_DEPTH,
    HP_OPTIMIZER,
    HP_LOSS,
    HP_BATCHNORM,
    HP_3PLUS,
    HP_BATCHSIZE,
    HP_VAL_BATCHSIZE,
    HP_LEARNING_RATE,
    HP_LOSS_WEIGHT,
    # HP_FSS_RADII,
    HP_L2_REG,
    HP_L1_REG
]

METRICS = ["binary_accuracy", "max_csi"]

METRICS_SUMMARY = [
    hp.Metric(
        "epoch_loss",
        group="validation",
        display_name="loss (val.)",
    ),
    hp.Metric(
        "epoch_loss",
        group="train",
        display_name="loss (train)",
    ),
    hp.Metric(
        "epoch_max_csi",
        group="validation",
        display_name="Max CSI (val.)",
    ),
    hp.Metric(
        "epoch_max_csi",
        group="train",
        display_name="Max CSI (train)",
    ),
    hp.Metric(
        "epoch_binary_accuracy",
        group="train",
        display_name="Bin Accuracy (train)",
    ),
    hp.Metric(
        "epoch_binary_accuracy",
        group="validation",
        display_name="Bin Accuracy (val.)",
    ),
]

def build_loss_dict(weight):# , FSS_radius):
    loss_dict = {}
    loss_dict['binary_crossentropy'] = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    loss_dict['weighted_binary_crossentropy'] = WeightedBinaryCrossEntropy(weights=[weight,1.0])
    # loss_dict['FSS'] = fractions_skill_score_loss(FSS_radius)
    return loss_dict

def build_metric_dict():
    metric_dict = {}
    metric_dict["binary_accuracy"] = tf.keras.metrics.BinaryAccuracy()
    metric_dict["max_csi"] = MaxCriticalSuccessIndex(scope=mirrored_strategy)
    if IS_3D_DATA:
        metric_dict["max_csi_0"] = MaxCriticalSuccessIndex(name="max_csi_0", scope=mirrored_strategy, is_3D=True, time_index=0)
        metric_dict["max_csi_15"] = MaxCriticalSuccessIndex(name="max_csi_15", scope=mirrored_strategy, is_3D=True, time_index=3)
        metric_dict["max_csi_30"] = MaxCriticalSuccessIndex(name="max_csi_30", scope=mirrored_strategy, is_3D=True, time_index=6)
        metric_dict["max_csi_45"] = MaxCriticalSuccessIndex(name="max_csi_45", scope=mirrored_strategy, is_3D=True, time_index=9)
        metric_dict["max_csi_55"] = MaxCriticalSuccessIndex(name="max_csi_55", scope=mirrored_strategy, is_3D=True, time_index=11)
    return metric_dict

def build_opt_dict(learning_rate):
    opt_dict = {}
    opt_dict['adam'] = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    opt_dict['adagrad'] = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
    opt_dict['sgd'] = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    opt_dict['rmsprop'] = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    return opt_dict

def model_fn(hparams, seed):
    """Create a Keras model with the given hyperparameters.
    Args:
      hparams: A dict mapping hyperparameters in `HPARAMS` to values.
      seed: A hashable object to be used as a random seed (e.g., to
        construct dropout layers in the model).
    Returns:
      A compiled Keras model.
    """
    rng = random.Random(seed)

    kernel_list = []
    for i in np.arange(1,hparams[HP_UNET_DEPTH]+1,1):
        kernel_list.append(hparams[HP_CONV_KERNELS]*i)

    if USE_MULTIPLE_GPUS:
        with mirrored_strategy.scope():
            # TODO: MAKE SURE TO MAKE COLLAPSE A SETTING THAT CAN BE CHANGED AT TOP
            if IS_3D_DATA:
                if hparams[HP_3PLUS]:
                    model = models.unet_3plus_3d(INPUT_SHAPE, kernel_list, n_labels=OUTPUT_CLASSES,kernel_size=hparams[HP_CONV_KERNEL_SIZE],
                                    stack_num_down=hparams[HP_CONV_LAYERS], stack_num_up=hparams[HP_CONV_LAYERS],
                                    activation=hparams[HP_CONV_ACTIVATION], output_activation=OUTPUT_ACTIVATION, weights=None,
                                    batch_norm=hparams[HP_BATCHNORM], l2=hparams[HP_L2_REG], l1=hparams[HP_L1_REG], pool='max', unpool='nearest', name='unet', collapse=False)
                else:
                    model = models.unet_3d(INPUT_SHAPE, kernel_list, n_labels=OUTPUT_CLASSES,kernel_size=hparams[HP_CONV_KERNEL_SIZE],
                                    stack_num_down=hparams[HP_CONV_LAYERS], stack_num_up=hparams[HP_CONV_LAYERS],
                                    activation=hparams[HP_CONV_ACTIVATION], output_activation=OUTPUT_ACTIVATION, weights=None,
                                    batch_norm=hparams[HP_BATCHNORM], l2=hparams[HP_L2_REG], l1=hparams[HP_L1_REG], pool='max', unpool='nearest', name='unet', collapse=False)
            else:
                if hparams[HP_3PLUS]:
                    model = models.unet_3plus_2d(INPUT_SHAPE, kernel_list, n_labels=OUTPUT_CLASSES,kernel_size=hparams[HP_CONV_KERNEL_SIZE],
                                    stack_num_down=hparams[HP_CONV_LAYERS], stack_num_up=hparams[HP_CONV_LAYERS],
                                    activation=hparams[HP_CONV_ACTIVATION], output_activation=OUTPUT_ACTIVATION, weights=None,
                                    batch_norm=hparams[HP_BATCHNORM], l2=hparams[HP_L2_REG], l1=hparams[HP_L1_REG], pool='max', unpool='nearest', name='unet')
                else:
                    model = models.unet_2d(INPUT_SHAPE, kernel_list, n_labels=OUTPUT_CLASSES,kernel_size=hparams[HP_CONV_KERNEL_SIZE],
                                    stack_num_down=hparams[HP_CONV_LAYERS], stack_num_up=hparams[HP_CONV_LAYERS],
                                    activation=hparams[HP_CONV_ACTIVATION], output_activation=OUTPUT_ACTIVATION, weights=None,
                                    batch_norm=hparams[HP_BATCHNORM], l2=hparams[HP_L2_REG], l1=hparams[HP_L1_REG], pool='max', unpool='nearest', name='unet')

            #compile losses: 
            loss_dict = build_loss_dict(hparams[HP_LOSS_WEIGHT])# , hparams[HP_FSS_RADII])
            opt_dict = build_opt_dict(hparams[HP_LEARNING_RATE])
            metric_dict = build_metric_dict()
            model.compile(
                loss=loss_dict[hparams[HP_LOSS]],
                optimizer=opt_dict[hparams[HP_OPTIMIZER]],
                metrics=list(itemgetter(*METRICS)(metric_dict)),
            )
    else:
        # TODO: MAKE SURE TO MAKE COLLAPSE A SETTING THAT CAN BE CHANGED AT TOP
        if IS_3D_DATA:
            if hparams[HP_3PLUS]:
                model = models.unet_3plus_3d(INPUT_SHAPE, kernel_list, n_labels=OUTPUT_CLASSES,kernel_size=hparams[HP_CONV_KERNEL_SIZE],
                                stack_num_down=hparams[HP_CONV_LAYERS], stack_num_up=hparams[HP_CONV_LAYERS],
                                activation=hparams[HP_CONV_ACTIVATION], output_activation=OUTPUT_ACTIVATION, weights=None,
                                batch_norm=hparams[HP_BATCHNORM], l2=hparams[HP_L2_REG], l1=hparams[HP_L1_REG], pool='max', unpool='nearest', name='unet', collapse=False)
            else:
                model = models.unet_3d(INPUT_SHAPE, kernel_list, n_labels=OUTPUT_CLASSES,kernel_size=hparams[HP_CONV_KERNEL_SIZE],
                                stack_num_down=hparams[HP_CONV_LAYERS], stack_num_up=hparams[HP_CONV_LAYERS],
                                activation=hparams[HP_CONV_ACTIVATION], output_activation=OUTPUT_ACTIVATION, weights=None,
                                batch_norm=hparams[HP_BATCHNORM], l2=hparams[HP_L2_REG], l1=hparams[HP_L1_REG], pool='max', unpool='nearest', name='unet', collapse=False)
        else:
            if hparams[HP_3PLUS]:
                model = models.unet_3plus_2d(INPUT_SHAPE, kernel_list, n_labels=OUTPUT_CLASSES,kernel_size=hparams[HP_CONV_KERNEL_SIZE],
                                stack_num_down=hparams[HP_CONV_LAYERS], stack_num_up=hparams[HP_CONV_LAYERS],
                                activation=hparams[HP_CONV_ACTIVATION], output_activation=OUTPUT_ACTIVATION, weights=None,
                                batch_norm=hparams[HP_BATCHNORM], l2=hparams[HP_L2_REG], l1=hparams[HP_L1_REG], pool='max', unpool='nearest', name='unet')
            else:
                model = models.unet_2d(INPUT_SHAPE, kernel_list, n_labels=OUTPUT_CLASSES,kernel_size=hparams[HP_CONV_KERNEL_SIZE],
                                stack_num_down=hparams[HP_CONV_LAYERS], stack_num_up=hparams[HP_CONV_LAYERS],
                                activation=hparams[HP_CONV_ACTIVATION], output_activation=OUTPUT_ACTIVATION, weights=None,
                                batch_norm=hparams[HP_BATCHNORM], l2=hparams[HP_L2_REG], l1=hparams[HP_L1_REG], pool='max', unpool='nearest', name='unet')

        #compile losses: 
        loss_dict = build_loss_dict(hparams[HP_LOSS_WEIGHT])# , hparams[HP_FSS_RADII])
        opt_dict = build_opt_dict(hparams[HP_LEARNING_RATE])
        metric_dict = build_metric_dict()
        model.compile(
            loss=loss_dict[hparams[HP_LOSS]],
            optimizer=opt_dict[hparams[HP_OPTIMIZER]],
            metrics=list(itemgetter(*METRICS)(metric_dict)),
        )

    trainable_count = np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])

    return model,trainable_count

def prepare_data(tf_ds_train_dir_glob, tf_ds_val_dir_glob):
    """ Load data """

    tf_ds_files = glob.glob(tf_ds_train_dir_glob)
    tf_ds_files.sort()
    tf_val_ds_files = glob.glob(tf_ds_val_dir_glob)
    tf_val_ds_files.sort()

    ######### TEMP #############
    # tf_ds_files = tf_ds_files[:32]
    # tf_val_ds_files = tf_val_ds_files[:32]
    ######################################

    complete_tf_ds = tf.data.experimental.load(tf_ds_files.pop(0))
    for tf_ds_file in tf_ds_files:
        tf_ds = tf.data.experimental.load(tf_ds_file)
        complete_tf_ds = complete_tf_ds.concatenate(tf_ds)

    complete_tf_ds_val = tf.data.experimental.load(tf_val_ds_files.pop(0))
    for tf_ds_file in tf_val_ds_files:
        tf_ds = tf.data.experimental.load(tf_ds_file)
        complete_tf_ds_val = complete_tf_ds_val.concatenate(tf_ds)
    
    if MEM_SAMPLES_NUM_IS_COMPLETE_DS_SIZE:
        complete_tf_ds = complete_tf_ds.cache()
        complete_tf_ds_val = complete_tf_ds_val.cache()
    
    return (complete_tf_ds, complete_tf_ds_val)

def run(data, base_logdir, session_id, hparams):
    """Run a training/validation session.
    Flags must have been parsed for this function to behave.
    Args:
      data: The data as loaded by `prepare_data()`.
      base_logdir: The top-level logdir to which to write summary data.
      session_id: A unique string ID for this session.
      hparams: A dict mapping hyperparameters in `HPARAMS` to values.
    """
    model,trainable_count = model_fn(hparams=hparams, seed=session_id)
    logdir = os.path.join(base_logdir, session_id)

    # if trainable_count > 4000000: #TODO: Maybe make this number a setting too?
    #     return

    ds_train, ds_val = data

    #batch the training data accordingly
    if MEM_SAMPLES_NUM_IS_COMPLETE_DS_SIZE:
        ds_train = ds_train.shuffle(NUM_SAMPLES_IN_MEM).batch(hparams[HP_BATCHSIZE])
    else:
        ds_train = ds_train.shuffle(NUM_SAMPLES_IN_MEM).repeat().batch(hparams[HP_BATCHSIZE])

    #this batch is arbitrary, just needed so that you dont overwelm RAM. 
    ds_val = ds_val.batch(hparams[HP_VAL_BATCHSIZE])

    callback = tf.keras.callbacks.TensorBoard(
        logdir,
        update_freq='epoch',
        profile_batch=0,  # workaround for issue #2084
    )

    hparams_callback = hp.KerasCallback(logdir, hparams)

    checkpoint_path = os.path.join(CHECKPOINTS_DIR, session_id)
    if IS_3D_DATA:
        # checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
        #                         monitor='val_max_csi_55', verbose=0, save_best_only=True, 
        #                         save_weights_only=False, save_freq='epoch', mode="max")
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                monitor='val_loss', verbose=0, save_best_only=True, 
                                save_weights_only=False, save_freq='epoch', mode="min")
    else:
        # checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
        #                         monitor='val_max_csi', verbose=0, save_best_only=True, 
        #                         save_weights_only=False, save_freq='epoch', mode="max")
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                monitor='val_loss', verbose=0, save_best_only=True, 
                                save_weights_only=False, save_freq='epoch', mode="min")
    
    callback_es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, mode="min")

    callback_loss = EarlyStoppingLoss()
    
    #add images to board 
    print(model.summary())

    try:
        if MEM_SAMPLES_NUM_IS_COMPLETE_DS_SIZE:
            result = model.fit(ds_train,
                epochs=flags.FLAGS.num_epochs,
                shuffle=True,
                validation_data=ds_val,
                validation_freq = VALIDATION_FREQ,
                callbacks=[callback, hparams_callback, checkpoint_callback, callback_es, callback_loss],verbose=0)
        else:
            result = model.fit(ds_train,
                epochs=flags.FLAGS.num_epochs,
                shuffle=True,
                validation_data=ds_val,
                validation_freq = VALIDATION_FREQ,
                steps_per_epoch = NUM_SAMPLES_IN_MEM // hparams[HP_BATCHSIZE],
                callbacks=[callback, hparams_callback, checkpoint_callback, callback_es, callback_loss],verbose=0)
    except:
        warnings.warn("Had to skip training a model because it raised an exception!")
        return

    #save trained model, need to build path first 
    # split_dir = logdir.split('log1')
    # right = split_dir[0][:-1] + split_dir[1]
    # left = H5_MODELS_DIR
    model_save_path = os.path.join(H5_MODELS_DIR, session_id + "model.h5")
    model.save(model_save_path)

    tf.keras.backend.clear_session()
    del model


def run_all(logdir, verbose=False):
    """Perform random search over the hyperparameter space.
    Arguments:
      logdir: The top-level directory into which to write data. This
        directory should be empty or nonexistent.
      verbose: If true, print out each run's name as it begins.
    """
    data = prepare_data(TF_TRAIN_DS_PATH_GLOB, TF_VAL_DS_PATH_GLOB)
    rng = random.Random(RANDOM_SEED)

    with tf.summary.create_file_writer(logdir).as_default():
        hp.hparams_config(hparams=HPARAMS, metrics=METRICS_SUMMARY)

    if RUN_SINGLE_MODEL:
        if type(RUN_START_INDEX) is not list:
            raise Exception("If RUN_SINGLE_MODEL is on, you must supply a list for RUN_START_INDEX")
    else:
        if type(RUN_START_INDEX) is not int:
            raise Exception("If RUN_SINGLE_MODEL is off, you must supply an int for RUN_START_INDEX")

    sessions_per_group = 1
    num_sessions = flags.FLAGS.num_session_groups * sessions_per_group
    session_index = 0  # across all session groups
    for group_index in range(flags.FLAGS.num_session_groups):
        hparams = {h: h.domain.sample_uniform(rng) for h in HPARAMS}
        hparams_string = str(hparams)
        if hparams[HP_3PLUS] and hparams[HP_UNET_DEPTH] < 3 and not RUN_SINGLE_MODEL:
            session_index += 1
            continue
        for repeat_index in range(sessions_per_group):
            session_id = str(session_index)
            if (not RUN_SINGLE_MODEL and session_index >= RUN_START_INDEX) or (RUN_SINGLE_MODEL and session_index in RUN_START_INDEX):
                if verbose:
                    print(
                        "--- Running training session %d/%d"
                        % (session_index, num_sessions)
                    )
                    print(hparams_string)
                    print("--- repeat #: %d" % (repeat_index + 1))
                run(
                    data=data,
                    base_logdir=logdir,
                    session_id=session_id,
                    hparams=hparams,
                )
            session_index += 1


def main(unused_argv):
    np.random.seed(RANDOM_SEED)
    logdir = flags.FLAGS.logdir
    # print('removing old logs')
    # shutil.rmtree(logdir, ignore_errors=True)
    print("Saving output to %s." % logdir)
    if IS_3D_DATA:
        METRICS.append("max_csi_0")
        METRICS.append("max_csi_15")
        METRICS.append("max_csi_30")
        METRICS.append("max_csi_45")
        METRICS.append("max_csi_55")
        METRICS_SUMMARY.append(hp.Metric(
            "epoch_max_csi_0",
            group="validation",
            display_name="Max CSI 0 (val.)",
            ))
        METRICS_SUMMARY.append(hp.Metric(
            "epoch_max_csi_0",
            group="train",
            display_name="Max CSI 0 (train)",
            ))
        METRICS_SUMMARY.append(hp.Metric(
            "epoch_max_csi_15",
            group="validation",
            display_name="Max CSI 15 (val.)",
            ))
        METRICS_SUMMARY.append(hp.Metric(
            "epoch_max_csi_15",
            group="train",
            display_name="Max CSI 15 (train)",
            ))
        METRICS_SUMMARY.append(hp.Metric(
            "epoch_max_csi_30",
            group="validation",
            display_name="Max CSI 30 (val.)",
            ))
        METRICS_SUMMARY.append(hp.Metric(
            "epoch_max_csi_30",
            group="train",
            display_name="Max CSI 30 (train)",
            ))
        METRICS_SUMMARY.append(hp.Metric(
            "epoch_max_csi_45",
            group="validation",
            display_name="Max CSI 45 (val.)",
            ))
        METRICS_SUMMARY.append(hp.Metric(
            "epoch_max_csi_45",
            group="train",
            display_name="Max CSI 45 (train)",
            ))
        METRICS_SUMMARY.append(hp.Metric(
            "epoch_max_csi_55",
            group="validation",
            display_name="Max CSI 55 (val.)",
            ))
        METRICS_SUMMARY.append(hp.Metric(
            "epoch_max_csi_55",
            group="train",
            display_name="Max CSI 55 (train)",
            ))
    run_all(logdir=logdir, verbose=True)
    print("Done. Output saved to %s." % logdir)


if __name__ == "__main__":
    if USE_MULTIPLE_GPUS:
        mirrored_strategy = tf.distribute.MirroredStrategy()
    else:
        mirrored_strategy = None

    app.run(main)
