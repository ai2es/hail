import xarray as xr
import numpy as np
import glob
from keras_unet_collection import models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from tensorflow import keras
import py3nvml


# TODO: When actually loading and working with WoFS data note that the ensemble changed over the years and
# this will have to be taken into account when creating train/val/testing datasets 
class Modeler:
    pass


if __name__ == "__main__":
    #number of GPUS I need:
    # TODO: Make controllable
    n_gpu = 1

    #find out how many are compltely free
    free_gpus = py3nvml.get_free_gpus()

    #count how many are free
    avail_gpu_ids = np.where(free_gpus)[0]

    #if there arent enough print it out
    if len(avail_gpu_ids) < n_gpu:
        print('Not enough GPUs, your job might fail')
    else:
        #if there are enough, the select the ones you need
        py3nvml.grab_gpus(num_gpus=n_gpu, gpu_select=[0])
    
    tf.config.threading.set_intra_op_parallelism_threads(16)
    tf.config.threading.set_inter_op_parallelism_threads(16)