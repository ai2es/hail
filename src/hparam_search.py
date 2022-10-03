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

import py3nvml

# Grab your prefered GPU
py3nvml.grab_gpus(num_gpus=1, gpu_select=[2])

import tensorflow as tf

# # GPU check
# physical_devices = tf.config.list_physical_devices('GPU') 
# n_physical_devices = len(physical_devices)
# if(n_physical_devices > 0):
#     for device in physical_devices:
#         tf.config.experimental.set_memory_growth(device, True)

import os
import random
import shutil
from absl import app
from absl import flags
import numpy as np
from tensorboard.plugins.hparams import api as hp
from keras_unet_collection import models
from operator import itemgetter
from custom_metrics import MaxCriticalSuccessIndex
import glob

if int(tf.__version__.split(".")[0]) < 2:
    # The tag names emitted for Keras metrics changed from "acc" (in 1.x)
    # to "accuracy" (in 2.x), so this demo does not work properly in
    # TensorFlow 1.x (even with `tf.enable_eager_execution()`).
    raise ImportError("TensorFlow 2.x is required to run this demo.")


flags.DEFINE_integer(
    "num_session_groups",
    300,
    "The approximate number of session groups to create.",
)
flags.DEFINE_string(
    "logdir",
    "/scratch/tgschmidt/temp_logdir",
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
    250,
    "Number of epochs per trial.",
)

# my params
TF_DS_PATH_GLOB = "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/trained_at_init_time/patches/train_val/tf_datasets/*"
H5_MODELS_DIR = "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/trained_at_init_time/saved_models/h5_models"
CHECKPOINTS_DIR = "/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/trained_at_init_time/saved_models/checkpoints"
# VAL_FRAC = 0.9 # Actually the train frac
NUM_SAMPLES_IN_MEM = 16000
INPUT_SHAPE = (64,64,15)
OUTPUT_CLASSES = 1
OUTPUT_ACTIVATION = "Sigmoid"
VALIDATION_FREQ = 5
# STEPS_PER_EPOCH = 20
PATIENCE = 5
# TF_DATASET_FILE_SAMPLE_NUM = 8000

#convolution params
HP_CONV_LAYERS = hp.HParam("conv_layers", hp.IntInterval(1, 3))
HP_CONV_KERNEL_SIZE = hp.HParam("conv_kernel_size", hp.Discrete([3, 5, 7]))
HP_CONV_ACTIVATION = hp.HParam("conv_activation", hp.Discrete(['LeakyReLU']))
HP_CONV_KERNELS = hp.HParam('num_of_kernels', hp.Discrete([4,8,16,32]))

#unet param
HP_UNET_DEPTH = hp.HParam('depth_of_unet', hp.Discrete([1,2,3,4]))
HP_OPTIMIZER = hp.HParam("optimizer", hp.Discrete(["adam"]))
HP_LOSS = hp.HParam("loss", hp.Discrete(["binary_crossentropy"])) 
HP_BATCHNORM = hp.HParam('batchnorm', hp.Discrete([False, True]))
HP_BATCHSIZE = hp.HParam('batch_size', hp.Discrete([32,64,128,256,512]))
HP_VAL_BATCHSIZE = hp.HParam('val_batch_size', hp.Discrete([5000]))
HP_LEARNING_RATE = hp.HParam('learning_rate', hp.Discrete([1e-2,1e-3]))

HPARAMS = [HP_CONV_LAYERS,
    HP_CONV_KERNEL_SIZE,
    HP_CONV_ACTIVATION,
    HP_CONV_KERNELS,
    HP_UNET_DEPTH,
    HP_OPTIMIZER,
    HP_LOSS,
    HP_BATCHNORM,
    HP_BATCHSIZE,
    HP_VAL_BATCHSIZE,
    HP_LEARNING_RATE,
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

def build_loss_dict(): #weight,thresh):
    loss_dict = {}
    loss_dict['binary_crossentropy'] = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    return loss_dict

def build_metric_dict():
    metric_dict = {}
    metric_dict["binary_accuracy"] = tf.keras.metrics.BinaryAccuracy()
    metric_dict["max_csi"] = MaxCriticalSuccessIndex()
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

    model = models.unet_2d(INPUT_SHAPE, kernel_list, n_labels=OUTPUT_CLASSES,kernel_size=hparams[HP_CONV_KERNEL_SIZE],
                      stack_num_down=hparams[HP_CONV_LAYERS], stack_num_up=hparams[HP_CONV_LAYERS],
                      activation=hparams[HP_CONV_ACTIVATION], output_activation=OUTPUT_ACTIVATION, weights=None,
                      batch_norm=hparams[HP_BATCHNORM], pool='max', unpool='nearest', name='unet')

    #compile losses: 
    loss_dict = build_loss_dict() #hparams[HP_LOSS_WEIGHT],hparams[HP_LOSS_THRESH])
    opt_dict = build_opt_dict(hparams[HP_LEARNING_RATE])
    metric_dict = build_metric_dict()
    model.compile(
        loss=loss_dict[hparams[HP_LOSS]],
        optimizer=opt_dict[hparams[HP_OPTIMIZER]],
        metrics=list(itemgetter(*METRICS)(metric_dict)),
    )
    return model

def prepare_data(tf_ds_dir_glob):
    """ Load data """

    #do this for both training and validations
    #load netcdf
    #convert to tensors ds_train = tf.data.Dataset.from_???_tensors(([125,125,8], [125,125,1]))
     
    #This is the tf.dataset route 
    # x_tensor_shape = (128, 128, 29)
    # y_tensor_shape = (128, 128, 1)
    # elem_spec = (tf.TensorSpec(shape=x_tensor_shape, dtype=tf.float16), tf.TensorSpec(shape=y_tensor_shape, dtype=tf.float16))

    # ds_train = tf.data.experimental.load('/scratch/randychase/updraft_training2.tf',
    #                                     elem_spec)

    # ds_val = tf.data.experimental.load('/scratch/randychase/updraft_validation2.tf',
    #                                     elem_spec)

    tf_ds_files = glob.glob(tf_ds_dir_glob)
    tf_ds_files.sort()

    # val_set_index = int(val_frac*len(tf_ds_files))
    # tf_val_ds_files = tf_ds_files[val_set_index:]
    # tf_ds_files = tf_ds_files[:val_set_index]

    complete_tf_ds = tf.data.experimental.load(tf_ds_files.pop(0))
    for tf_ds_file in tf_ds_files:
        tf_ds = tf.data.experimental.load(tf_ds_file)
        complete_tf_ds = complete_tf_ds.concatenate(tf_ds)

    complete_tf_ds_val = tf.data.experimental.load(tf_val_ds_files.pop(0))
    for tf_ds_file in tf_val_ds_files:
        tf_ds = tf.data.experimental.load(tf_ds_file)
        complete_tf_ds_val = complete_tf_ds_val.concatenate(tf_ds)
    
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
    model = model_fn(hparams=hparams, seed=session_id)
    logdir = os.path.join(base_logdir, session_id)

    ds_train, ds_val = data

    #batch the training data accordingly
    ds_train = ds_train.shuffle(NUM_SAMPLES_IN_MEM).repeat().batch(hparams[HP_BATCHSIZE]) #TODO: Remove .repeat?

    #this batch is arbitrary, just needed so that you dont overwelm RAM. 
    ds_val = ds_val.batch(hparams[HP_VAL_BATCHSIZE])

    callback = tf.keras.callbacks.TensorBoard(
        logdir,
        update_freq='epoch',
        profile_batch=0,  # workaround for issue #2084
    )

    hparams_callback = hp.KerasCallback(logdir, hparams)

    checkpoint_path = os.path.join(CHECKPOINTS_DIR, session_id)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                            monitor='val_max_csi', verbose=0, save_best_only=True, 
                            save_weights_only=False, save_freq="epoch", mode="max")
    
    callback_es = tf.keras.callbacks.EarlyStopping(monitor='val_max_csi', patience=PATIENCE, mode="max")
    
    #add images to board 
    print(model.summary())

    result = model.fit(ds_train,
        epochs=flags.FLAGS.num_epochs,
        shuffle=False,
        validation_data=ds_val,
        validation_freq = VALIDATION_FREQ,
        steps_per_epoch = NUM_SAMPLES_IN_MEM // hparams[HP_BATCHSIZE], 
        callbacks=[callback, hparams_callback, checkpoint_callback, callback_es],verbose=1)

    #save trained model, need to build path first 
    # split_dir = logdir.split('log1')
    # right = split_dir[0][:-1] + split_dir[1]
    # left = H5_MODELS_DIR
    model_save_path = os.path.join(H5_MODELS_DIR, session_id + "model.h5")
    model.save(model_save_path)


def run_all(logdir, verbose=False):
    """Perform random search over the hyperparameter space.
    Arguments:
      logdir: The top-level directory into which to write data. This
        directory should be empty or nonexistent.
      verbose: If true, print out each run's name as it begins.
    """
    data = prepare_data(TF_DS_PATH_GLOB)
    rng = random.Random(0)

    with tf.summary.create_file_writer(logdir).as_default():
        hp.hparams_config(hparams=HPARAMS, metrics=METRICS_SUMMARY)

    sessions_per_group = 1
    num_sessions = flags.FLAGS.num_session_groups * sessions_per_group
    session_index = 0  # across all session groups
    for group_index in range(flags.FLAGS.num_session_groups):
        hparams = {h: h.domain.sample_uniform(rng) for h in HPARAMS}
        hparams_string = str(hparams)
        for repeat_index in range(sessions_per_group):
            session_id = str(session_index)
            session_index += 1
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


def main(unused_argv):
    np.random.seed(0)
    logdir = flags.FLAGS.logdir
    print('removing old logs')
    shutil.rmtree(logdir, ignore_errors=True)
    print("Saving output to %s." % logdir)
    run_all(logdir=logdir, verbose=True)
    print("Done. Output saved to %s." % logdir)


if __name__ == "__main__":
    app.run(main)