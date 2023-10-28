import numpy as np

from deepthink.layers.pooling.base_pooling import BasePooling
from deepthink.utils import get_strided_view_2D


class AveragePooling2D(BasePooling):
    """
    Average pooling operation for 2D data.

    Downsamples input data by taking the mean value from each
    spatial pooling window. When pooling window size and stride are
    both 2 (default values) the resulting array is halved along
    height and width.

    Input shape should be (batch, channels, height, width).

    Parameters
    ----------
    pool_size : int
        The size of the pooling window.
    stride : int
        The step size between each pooling window.

    Attributes
    ----------
    scaling_factor : int
        The factor by which the output is scaled. This is equal
        to the number of elements in the pooling window.
    """
    def __init__(self, pool_size=2, stride=2, **kwargs):
        super().__init__(
            pool_size=pool_size,
            stride=stride,
            **kwargs
        )
        # Calculate scaling factor for mean pooling backprop
        self.scaling_factor = self.pool_size**2

    def __repr__(self):
        return 'AveragePooling2D'

    def initialize(self):
        """
        Initialize settings to prepare the layer for training.
        """
        # Get input shape and calculate output shape
        self.set_output_size()

        self.output = np.zeros((self.batch_size, self.n_channels,
                                self.output_size, self.output_size),
                               dtype=self.dtype)
        # Create the shapes to use with "get_strided_view"
        self.forward_view_shape = (self.batch_size, self.output_size,
                                   self.output_size, self.n_channels,
                                   self.pool_size, self.pool_size)

    def forward(self, inputs):
        """
        Perform one forward pass of average pooling operation.

        Parameters
        ----------
        inputs : np.array
            Input tensor to perform forward pass on, with shape:
            (batch_size, channels, img_size_in, img_size_in)

        Returns
        -------
        output : np.array
            Average-pooled output tensor with shape:
            (batch_size, channels, img_size_out, img_size_out)
        """
        view = get_strided_view_2D(inputs,
                                   self.forward_view_shape,
                                   self.stride)

        self.output = np.mean(view, axis=(4, 5))
        self.output = self.output.transpose(0, 3, 1, 2)
        return self.output

    def backward(self, grads):
        """
        Perform backward pass.

        Calculates partial derivatives w.r.t. inputs and stores them
        in the `dinputs` instance variable.

        Parameters
        ----------
        grads : np.array
            Gradients of the subsequent layer.

        Returns
        -------
        dinputs : np.array
            Gradients of the inputs.
        """
        self.dinputs = np.tile(grads / self.scaling_factor,
                               self.input_shape)
        return self.dinputs
