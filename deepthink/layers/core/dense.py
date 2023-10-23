import numpy as np

from deepthink.utils import initialize_weights
from deepthink.layers.layer import BaseLayer


class Dense(BaseLayer):
    """
    Fully-connected NN layer.

    During the forward pass it performs matrix multiplication between
    the inputs and weights with addition of bias as an offset.
    Backward pass computes derivatives for weights, bias and inputs.

    Parameters
    ----------
    n_neurons : int
        The number of neurons, dimensionality of output space.
    input_shape : tuple,default=None
        Input shape. Required if first layer, otherwise calculated
        during initialization. Should have shape (N, D).
    """
    def __init__(self, n_neurons, input_shape=None, **kwargs):
        super().__init__(**kwargs)
        self.n_neurons = n_neurons
        self.input_shape = input_shape

    def __str__(self):
        return f'Dense({self.n_neurons})'

    def initialize(self):
        """
        Initialize settings to prepare the layer for training
        """
        # If layer is not first layer use prev_layer to get input_shape
        if self.prev_layer:
            self.input_shape = self.prev_layer.output.shape
        elif self.prev_layer is None and self.input_shape is None:
            raise ValueError(
                'input_shape is a required argument for the first layer.')

        # Initialize output array, shape: (batch-size, n_neurons)
        self.output = np.zeros((self.input_shape[0], self.n_neurons),
                               dtype=self.dtype)
        # Initialize weights and biases.
        # Weights have shape: (input-dimensions, output-dimensions)
        weights_shape = (self.input_shape[-1], self.n_neurons)
        self.weights = initialize_weights(weights_shape,
                                          self.weight_init,
                                          self.dtype)
        self.bias = np.zeros((1, self.n_neurons),
                             dtype=self.dtype)
        # Initialize arrays to store optimizer momentum values
        self.weight_momentum = np.zeros(weights_shape,
                                        dtype=self.dtype)
        self.bias_momentum = np.zeros(self.bias.shape,
                                      dtype=self.dtype)
        # Initialize arrays to store optimizer gradient cache
        self.weight_grad_cache = np.zeros(weights_shape,
                                          dtype=self.dtype)
        self.bias_grad_cache = np.zeros(self.bias.shape,
                                        dtype=self.dtype)

    def forward(self, inputs):
        """
        Perform one forward pass on inputs.
        """
        # Store input to be used in backpropagation
        self.input = inputs
        self.output = np.dot(inputs, self.weights) + self.bias
        return self.output

    def backward(self, grads):
        """
        Perform backpropagation by computing partial
        derivatives for weights, bias and inputs.
        """
        self.dweights = np.dot(self.input.T, grads)
        self.dbiases = np.sum(grads, axis=0, keepdims=True)
        self.dinputs = np.dot(grads, self.weights.T)
