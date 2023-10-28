import numpy as np

from deepthink.utils import initialize_weights
from deepthink.layers.layer import BaseLayer


class Flatten(BaseLayer):
    """
    Flattens the input array.

    Batch-size is the only dimension preserved. Used to
    reshape output from convolution/pooling layers to be
    used as input for a Dense layer.
    """
    def __init__(self, input_shape=None, **kwargs):
        super().__init__(**kwargs)
        self.input_shape = input_shape

    def initialize(self):
        """
        Initialize settings to prepare the layer for training
        """
        if self.prev_layer is None and self.input_shape is None:
            raise ValueError('Flatten cannot be the first layer')

        if self.input_shape is None:
            self.input_shape = self.prev_layer.output.shape

        self.output = np.zeros(
            self.input_shape,
            dtype=self.dtype).reshape((self.input_shape[0], -1))

    def forward(self, inputs):
        """
        Perform one forward pass by flattening input array.
        """
        self.output = inputs.ravel().reshape(self.output.shape)
        return self.output

    def backward(self, grads):
        """
        Perform backward pass.

        Gradients are reshaped from 1D to their
        input dimensions.
        """
        self.dinputs = grads.reshape(self.input_shape)
