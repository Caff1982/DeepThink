import numpy as np


class BaseLayer:
    """
    Base layer class from which all layers inherit.

    A layer is an object that takes an input tensor and performs some
    operation on it, using the "forward" method. It also has a
    "backward" method for calculating derivatives as part of
    backpropagation. Calling 'initialization' is required to set
    prev_layer, next_layer, etc.

    Layers are designed to be chained together in a list which is an
    instance attribute of the Model class. One forward pass can then
    be completed by calling 'forward' on each subsequent layer and
    obtaining the output. Layers can also be used individually where
    required.

    Parameters
    ----------
    prev_layer : (any subclass of BaseLayer)
        The previous layer in the neural network. Added automatically
        during initialization.
    next_layer : (any subclass of BaseLayer)
        The next layer in the neural network. Added automatically
        during initialization.
    input_shape : tuple
        The shape of the input array. This is a required argument
        for the first layer, otherwise computed during
        initialization. Should have shape (N, D) or (N, D, H, W).
    weight_init : str
        The type of initialization to use when creating the layer
        weights. See initialize_weights in utils for more info.
    dtype : type
        The numpy datatype to be used. Uses np.float32 by default,
        np.float64 is required for gradient checking.
    """
    def __init__(self, prev_layer=None, next_layer=None,
                 input_shape=None, weight_init='he_uniform',
                 dtype=np.float32):
        self.prev_layer = prev_layer
        self.next_layer = next_layer
        self.input_shape = input_shape
        self.weight_init = weight_init
        self.dtype = dtype

    def initialize(self):
        raise NotImplementedError(
            'All BaseLayer subclasses must implement initialize method'
        )

    def forward(self, X):
        raise NotImplementedError(
            'All BaseLayer subclasses must implement forward method'
        )

    def backward(self, grads):
        raise NotImplementedError(
            'All BaseLayer subclasses must implement backward method'
        )
