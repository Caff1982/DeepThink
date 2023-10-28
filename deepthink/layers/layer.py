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
    prev_layer : (any subclass of BaseLayer), default=None
        The previous layer in the neural network. Added automatically
        during initialization.
    next_layer : (any subclass of BaseLayer), default=None
        The next layer in the neural network. Added automatically
        during initialization.
    input_shape : tuple, default=None
        The shape of the input array. This is a required argument
        for the first layer, otherwise computed during
        initialization. Should have shape (N, D) or (N, D, H, W).
    weight_init : str, default='he_uniform'
        The type of initialization to use when creating the layer
        weights. See initialize_weights in utils for more info.
    dtype : type, default=np.float32
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

        # Create placehoders to be defined in initialize method
        self.input = None
        self.output = None
        self.weight_momentum = None
        self.bias_momentum = None
        self.weight_grad_cache = None
        self.bias_grad_cache = None

    @property
    def input_shape(self):
        # If the _input_shape is not set, infer it from the previous layer
        if self._input_shape is None:
            if self.prev_layer is None:
                raise ValueError(
                    'input_shape is a required argument for the first layer.'
                )
            return self.prev_layer.output.shape
        return self._input_shape

    @input_shape.setter
    def input_shape(self, shape):
        self._input_shape = shape

    def __repr__(self):
        return f'{self.__class__.__name__}'

    def initialize(self):
        raise NotImplementedError(
            'All BaseLayer subclasses must implement initialize method'
        )

    def forward(self, inputs):
        raise NotImplementedError(
            'All BaseLayer subclasses must implement forward method'
        )

    def backward(self, grads):
        raise NotImplementedError(
            'All BaseLayer subclasses must implement backward method'
        )
