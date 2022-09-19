import tensorflow as tf 
import tensorflow.keras.backend as K
import numpy as np


class MaxCriticalSuccessIndex(tf.keras.metrics.Metric):
    """ 
    Calcualte the element-wise [e.g., pixel-wise],
    maximum Critical Success Index out of 20 thresholds.
    If update_state is called more than once, it will add give you the new max 
    csi over all calls. In other words, it accumulates the TruePositives, 
    FalsePositives and FalseNegatives. 
    
    If you want to use it on a fresh y_true,y_pred, make sure you run:

    metric.reset_state()

    then 

    metric(y_true,y_pred) #<-- this will also store the counts of TP etc.

    The shapes of y_true and y_pred should match 
    
    """ 

    def __init__(self, name="max_csi",
                 tp = tf.keras.metrics.TruePositives(thresholds=np.arange(0.05,1.05,0.05).tolist()),
                 fp = tf.keras.metrics.FalsePositives(thresholds=np.arange(0.05,1.05,0.05).tolist()),
                 fn = tf.keras.metrics.FalseNegatives(thresholds=np.arange(0.05,1.05,0.05).tolist()),
                 **kwargs):
        super(MaxCriticalSuccessIndex, self).__init__(name=name, **kwargs)

        #initialize csi value, if no data given, it will be 0 
        self.csi = self.add_weight(name="csi", initializer="zeros")

        #store defined metric functions
        self.tp = tp 
        self.fp = fp 
        self.fn = fn
        #flush functions just in case 
        self.tp.reset_state()
        self.fp.reset_state()
        self.fn.reset_state()

    def update_state(self, y_true, y_pred, sample_weight=None):
        # TODO: Make sure this supports 1-hot encoding? I think it does? <---------------
        if (len(y_true.shape[1:]) > 2) and (y_true.shape[-1] == 1):
            y_true = y_true[:,:,:,0]
            y_pred = y_pred[:,:,:,0]
            #ravel for pixelwise comparison
            y_true = tf.experimental.numpy.ravel(y_true)
            y_pred = tf.experimental.numpy.ravel(y_pred)
        #if the output is a map (batch,nx,ny,nl) ravel it
        elif (len(y_true.shape[1:]) > 2) and (y_true.shape[-1] > 1) :
            raise Exception("Max CSI not supported for non-binary rn")

        #call vectorized stats metrics, add them to running amount of each
        self.tp.update_state(y_true,y_pred)
        self.fp.update_state(y_true,y_pred)
        self.fn.update_state(y_true,y_pred)

        #calc current max csi (so we can get updates per batch)
        self.csi_val = tf.reduce_max(self.tp.result()/(self.tp.result() + self.fn.result() + self.fp.result()))

        #assign the value to the csi 'weight'
        self.csi.assign(self.csi_val)
      
    def result(self):
        return self.csi

    def reset_state(self):
        # Reset the counts 
        self.csi.assign(0.0)
        self.tp.reset_state()
        self.fp.reset_state()
        self.fn.reset_state()
