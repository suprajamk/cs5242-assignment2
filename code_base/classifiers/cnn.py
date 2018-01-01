from builtins import object
import numpy as np

from code_base.layers import *
from code_base.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, dropout=0, seed=123, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.use_dropout = dropout > 0
    self.dtype = dtype
    
    C, H, W = input_dim
    self.params['W1'] = np.random.normal(0, weight_scale, (num_filters, C, filter_size, filter_size))
    self.params['b1'] = np.zeros(num_filters)
    self.params['W2'] = np.random.normal(0, weight_scale, (int(num_filters*H/2*W/2), hidden_dim))
    self.params['b2'] = np.zeros(hidden_dim)
    self.params['W3'] = np.random.normal(0, weight_scale, (hidden_dim, num_classes))
    self.params['b3'] = np.zeros(num_classes)
    self.dropout_param = {}
    
    if self.use_dropout:
        self.dropout_param = {'mode': 'train', 'p': dropout}
        if seed is not None:
            self.dropout_param['seed'] = seed
    
    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    mode = 'test' if y is None else 'train'
    
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1)}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    # Set train/test mode for dropout param since it
    # behaves differently during training and testing.
    if self.use_dropout:
        self.dropout_param['mode'] = mode
    
    scores = None
    conv_out, conv_cache = conv_forward(X, W1, b1, conv_param)
    relu1_out, relu1_cache = relu_forward(conv_out)
    pool_out, pool_cache = max_pool_forward(relu1_out, pool_param)
    affine_relu_out, affine_relu_cache = affine_relu_forward(pool_out, W2, b2)
    if self.use_dropout:
            affine_relu_out, dropout_cache = dropout_forward(affine_relu_out, self.dropout_param)
    affine2_out, affine2_cache = affine_forward(affine_relu_out, W3, b3)
    scores = affine2_out
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    loss, dscores = softmax_loss(scores, y)
    loss += 0.5 * self.reg*(np.sum(self.params['W1']* self.params['W1']) 
         + np.sum(self.params['W2']* self.params['W2'])+np.sum(self.params['W3']* self.params['W3']))
    
    affine2_dx, affine2_dw, affine2_db = affine_backward(dscores, affine2_cache)
    if self.use_dropout:
            affine2_dx = dropout_backward(affine2_dx, dropout_cache)
            
    grads['W3'] = affine2_dw + self.reg * self.params['W3']
    grads['b3'] = affine2_db
       
    affine1_dx, affine1_dw, affine1_db = affine_relu_backward(affine2_dx, affine_relu_cache)
    grads['W2'] = affine1_dw + self.reg * self.params['W2']
    grads['b2'] = affine1_db

    pool_dx = max_pool_backward(affine1_dx, pool_cache)
    relu_dx = relu_backward(pool_dx, relu1_cache)
    conv_dx, conv_dw, conv_db = conv_backward(relu_dx, conv_cache)
    grads['W1'] = conv_dw + self.reg * self.params['W1']
    grads['b1'] = conv_db
    
    return loss, grads
  
  
pass
