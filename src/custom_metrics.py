import tensorflow as tf 
import tensorflow.keras.backend as K
import numpy as np
from tensorflow.python.framework import constant_op


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
                #  tp = tf.keras.metrics.TruePositives(thresholds=np.arange(0.05,1.05,0.05).tolist()),
                #  fp = tf.keras.metrics.FalsePositives(thresholds=np.arange(0.05,1.05,0.05).tolist()),
                #  fn = tf.keras.metrics.FalseNegatives(thresholds=np.arange(0.05,1.05,0.05).tolist()),
                 is_3D = False,
                 time_index = 0,
                 scope = None,
                 **kwargs):

        if scope is not None:
            with scope.scope():
                tp = tf.keras.metrics.TruePositives(thresholds=np.arange(0.05,1.05,0.05).tolist())
                fp = tf.keras.metrics.FalsePositives(thresholds=np.arange(0.05,1.05,0.05).tolist())
                fn = tf.keras.metrics.FalseNegatives(thresholds=np.arange(0.05,1.05,0.05).tolist())
        else:
            tp = tf.keras.metrics.TruePositives(thresholds=np.arange(0.05,1.05,0.05).tolist())
            fp = tf.keras.metrics.FalsePositives(thresholds=np.arange(0.05,1.05,0.05).tolist())
            fn = tf.keras.metrics.FalseNegatives(thresholds=np.arange(0.05,1.05,0.05).tolist())

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

        self.is_3D = is_3D
        self.time_index = time_index

    def update_state(self, y_true, y_pred, sample_weight=None):
        if len(y_true.shape) > 1:
            if y_true.shape[-1] > 1:
                raise Exception("Max CSI not supported for non-binary rn")
            if self.is_3D:
                y_true = y_true[...,self.time_index,0]
                y_pred = y_pred[...,self.time_index,0]
            else:
                y_true = y_true[...,0]
                y_pred = y_pred[...,0]
            
        #ravel for pixelwise comparison
        y_true = tf.experimental.numpy.ravel(y_true)
        y_pred = tf.experimental.numpy.ravel(y_pred)
        #if the output is a map (batch,nx,ny,nl) ravel it

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


def fractions_skill_score_loss(mask_size, c=1.0, cutoff=0.5, want_hard_discretization=False):
    """
    Make fractions skill score loss function. Visit https://github.com/CIRA-ML/custom_loss_functions for documentation.
    Parameters
    ----------
    mask_size: int or tuple
        - Size of the mask/pool in the AveragePooling layers.
    c: int or float
        - C parameter in the sigmoid function. This will only be used if 'want_hard_discretization' is False.
    cutoff: float
        - If 'want_hard_discretization' is True, y_true and y_pred will be discretized to only have binary values (0/1)
    want_hard_discretization: bool
        - If True, y_true and y_pred will be discretized to only have binary values (0/1).
        - If False, y_true and y_pred will be discretized using a sigmoid function.
    Returns
    -------
    fractions_skill_score: float
        - Fractions skill score.
    """

    pool_kwargs = {'pool_size': mask_size}

    pool1 = tf.keras.layers.AveragePooling2D(**pool_kwargs)
    pool2 = tf.keras.layers.AveragePooling2D(**pool_kwargs)

    @tf.function
    def fractions_skill_score(y_true, y_pred):
        """
        Fractions skill score (FSS) loss function.
        y_true: tf.Tensor
            - The image that the model is trying to predict.
        y_pred: tf.Tensor
            - The model's actual prediction.
        """

        # TODO: Test if below correct
        # If the tensor represents a 3D image, turn it into a 2D image by taking the maximum along the vertical columns
        if len(y_pred.shape) == 5:
            y_true = tf.experimental.numpy.mean(y_true, axis=3) # Axis was 2 for some reason and was also amax instead
            y_pred = tf.experimental.numpy.mean(y_pred, axis=3) # Axis was 2 for some reason and was also amax instead

        if want_hard_discretization:
            y_true_binary = tf.where(y_true > cutoff, 1.0, 0.0)
            y_pred_binary = tf.where(y_pred > cutoff, 1.0, 0.0)
        else:
            y_true_binary = tf.math.sigmoid(c * (y_true - cutoff))
            y_pred_binary = tf.math.sigmoid(c * (y_pred - cutoff))

        y_true_density = pool1(y_true_binary)
        n_density_pixels = tf.cast((tf.shape(y_true_density)[1] * tf.shape(y_true_density)[2]), tf.float32)

        y_pred_density = pool2(y_pred_binary)

        MSE_n = tf.keras.metrics.mean_squared_error(y_true_density, y_pred_density)

        O_n_squared_image = tf.keras.layers.Multiply()([y_true_density, y_true_density])
        O_n_squared_vector = tf.keras.layers.Flatten()(O_n_squared_image)
        O_n_squared_sum = tf.reduce_sum(O_n_squared_vector)

        M_n_squared_image = tf.keras.layers.Multiply()([y_pred_density, y_pred_density])
        M_n_squared_vector = tf.keras.layers.Flatten()(M_n_squared_image)
        M_n_squared_sum = tf.reduce_sum(M_n_squared_vector)

        MSE_n_ref = (O_n_squared_sum + M_n_squared_sum) / n_density_pixels

        my_epsilon = tf.keras.backend.epsilon()  # this is 10^(-7)

        if want_hard_discretization:
            if MSE_n_ref == 0:
                return MSE_n
            else:
                return MSE_n / MSE_n_ref
        else:
            return MSE_n / (MSE_n_ref + my_epsilon)

    return fractions_skill_score


def max_FSS_test_dataset(full_obs, full_fcst, radius):
    # Expects (n_samples, lat, lon) shapes

    thresholds = np.arange(0.05,1.05,0.05)

    all_poss_FSS_outcomes = []
    for thres in thresholds:
        FSS_outcome = np.mean([compute_FSS_metric(full_obs[i,:,:], full_fcst[i,:,:], radius, thres) for i in range(full_obs.shape[0])])
        all_poss_FSS_outcomes.append(FSS_outcome)

    return np.max(all_poss_FSS_outcomes)


def compute_FSS_metric(obs, fcst, nbrhd_rad, thres, buf_width=0, min_points=30):

  # From Dr. Corey Potvin
  # Compute FSS for a single threshold
  # Required inputs: 2D obs/truth field ('obs'), 2D model/analysis field ('fcst'), FSS scale ('nbrhd_rad'), FSS threshold ('thres')
  # Optional inputs: FSS computations excluded from within 'buf_width' of domain boundary; FSS not computed if number of obs and fcst points exceeding 'thres' is less than 'min_points'

  total=0; total2=0; count=0
  nbrhd_rad = int(nbrhd_rad)
  buf_width = int(buf_width)
  if buf_width==0:
    buf_width = nbrhd_rad

  # Exclude buffer zone from calculations of obs_points, fcst_points
  obs_domain = obs[buf_width:-buf_width, buf_width:-buf_width]
  fcst_domain = fcst[buf_width:-buf_width, buf_width:-buf_width]
  # Compute number of obs and fcst points exceeding FSS intensity threshold
  obs_points = obs_domain[obs_domain>=thres].size
  fcst_points = fcst_domain[fcst_domain>=thres].size

  if obs_points < min_points and fcst_points < min_points:
    return -999

  # Iterate through FSS computational domain    
  for i in range(buf_width, obs.shape[0]-buf_width):
    for j in range(buf_width, obs.shape[1]-buf_width):
      # Get neighborhood centered on current grid point
      obs_nbrhd = obs[i-nbrhd_rad:i+nbrhd_rad+1,j-nbrhd_rad:j+nbrhd_rad+1]
      fcst_nbrhd = fcst[i-nbrhd_rad:i+nbrhd_rad+1,j-nbrhd_rad:j+nbrhd_rad+1]
      # Compute neighborhood probabilities of exceeding intensity threshold 
      NP_obs = obs_nbrhd[obs_nbrhd>=thres].size/obs_nbrhd.size
      NP_fcst = fcst_nbrhd[fcst_nbrhd>=thres].size/fcst_nbrhd.size
      total += (NP_fcst-NP_obs)**2 # accumulate squared differences between obs and fcst neighborhood probabilities
      total2 += (NP_fcst**2+NP_obs**2) # accumulate worst possible squared differences between obs and fcst neighborhood probabilities
      count += 1

  if count==0:
    print ('FSS domain is zero gridpoints! Is buf_width too large?')
    return 0

  FBS = total/count
  FBS_worst = total2/count

  if FBS_worst==0:
    FSS = 1
  else:
    FSS = 1 - FBS/FBS_worst

  return FSS


class WeightedBinaryCrossEntropy(tf.keras.losses.Loss):
    """Adapted tf.keras.backend.binary_crossentropy to have weights that work for a UNET
    
    Last I checked, weighting the binary cross entropy wouldnt work with the UNETs, it
    use to complain about the shape being not right. 
    
    To get around this, the class here has a way to weight the 1s and 0s differently.
    This is important for alot of meteorological instances where the event we are looking 
    to predict is often a 'rare' event. We mean where the pixels == 1 are like 1% of the 
    total number of pixels. 
    
    the expected shape of y_true is [n_sample,nx,ny,1]
    
    """
    def __init__(self,weights=[1.0,1.0],from_logits=False):
        super().__init__()
        #store weights
        self.w1 = weights[0]
        self.w2 = weights[1]
        self.from_logits=from_logits

    def __str__(self):
        return ("WeightedBinaryCrossEntropy()")
    
    def binary_crossentropy(self,target, output):
        """Binary crossentropy between an output tensor and a target tensor.
        Args:
          target: A tensor with the same shape as `output`.
          output: A tensor.
          from_logits: Whether `output` is expected to be a logits tensor.
              By default, we consider that `output`
              encodes a probability distribution.
        Returns:
          A tensor.
        """
        target = tf.convert_to_tensor(target)
        output = tf.convert_to_tensor(output)

        # Use logits whenever they are available. `softmax` and `sigmoid`
        # activations cache logits on the `output` Tensor.
        if hasattr(output, '_keras_logits'):
            output = output._keras_logits  # pylint: disable=protected-access
            if self.from_logits:
              warnings.warn(
                  '"`binary_crossentropy` received `from_logits=True`, but the `output`'
                  ' argument was produced by a sigmoid or softmax activation and thus '
                  'does not represent logits. Was this intended?"',
                  stacklevel=2)
            self.from_logits = True

        if self.from_logits:
            return tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=output)

        if (not isinstance(output, (tf.__internal__.EagerTensor, tf.Variable)) and
          output.op.type == 'Sigmoid') and not hasattr(output, '_keras_history'):
            # When sigmoid activation function is used for output operation, we
            # use logits from the sigmoid function directly to compute loss in order
            # to prevent collapsing zero when training.
            assert len(output.op.inputs) == 1
            output = output.op.inputs[0]
            return tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=output)

        epsilon_ = _constant_to_tensor(tf.keras.backend.epsilon(), output.dtype.base_dtype)
        output = tf.clip_by_value(output, epsilon_, 1. - epsilon_)

        # Compute cross entropy from probabilities.
        bce = target * tf.math.log(output + tf.keras.backend.epsilon())
        bce += (1 - target) * tf.math.log(1 - output + tf.keras.backend.epsilon()) 
        return -bce

    def call(self, y_true, y_pred):
        
        y_true = tf.cast(y_true, tf.float64)
        y_pred = tf.cast(y_pred, tf.float64)
        
        loss = self.binary_crossentropy(y_true, y_pred)
    
        #build weight_matrix 
        ones_array = tf.ones_like(y_true)
        weights_for_nonzero = tf.math.multiply(ones_array,self.w1)
        weights_for_zero = tf.math.multiply(ones_array,self.w2)
        weight_matrix = tf.where(tf.greater(y_true,0.0),weights_for_nonzero,
                           weights_for_zero)
        self.w = weight_matrix
        #weight non-zero pixels more
        loss = tf.math.multiply(weight_matrix,loss)
    
        return tf.reduce_mean(loss)

def _constant_to_tensor(x, dtype):
    """Convert the input `x` to a tensor of type `dtype`.
    This is slightly faster than the _to_tensor function, at the cost of
    handling fewer cases.
    Args:
        x: An object to be converted (numpy arrays, floats, ints and lists of
            them).
        dtype: The destination type.
    Returns:
        A tensor.
    """
    return constant_op.constant(x, dtype=dtype)