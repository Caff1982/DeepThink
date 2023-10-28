import numpy as np

from deepthink.layers.pooling.base_pooling import BasePooling
from deepthink.utils import get_strided_view_2D


class MaxPooling2D(BasePooling):
    """
    Max pooling operation for 2D data.

    Downsamples input data by taking the maximum value from each
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
    """
    def __init__(self, pool_size=2, stride=2, **kwargs):
        super().__init__(
            pool_size=pool_size,
            stride=stride,
            **kwargs
        )

    def initialize(self):
        """
        Initialize settings to prepare the layer for training
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
        Perform one forward pass of MaxPooling operation.

        Parameters
        ----------
        inputs : np.array
            Input tensor to perform forward pass on, with shape:
            (batch_size, channels, img_size_in, img_size_in)

        Returns
        -------
        output : np.array
            Max-pooled output tensor with shape:
            (batch_size, channels, img_size_out, img_size_out)
        """
        view = get_strided_view_2D(inputs,
                                   self.forward_view_shape,
                                   self.stride)

        self.output = np.max(view, axis=(4, 5), keepdims=True)
        # Create a mask of maximal values to use in backprop
        self.max_args = np.where(self.output == view, 1, 0)
        self.output = np.squeeze(self.output, axis=(4, 5))
        self.output = self.output.transpose(0, 3, 1, 2)
        return self.output

    def backward(self, grads):
        """
        Perform backward pass.

        Calculates partial derivatives w.r.t. inputs and stores them in
        the `dinputs` instance variable.

        Parameters
        ----------
        grads : np.array
            Gradients of the subsequent layer.

        Returns
        -------
        dinputs : np.array
            Gradients of the inputs.
        """
        # Initialize empty array
        self.dinputs = np.zeros(self.input_shape)

        # Use max_args mask to get the maximal indices
        im, ih, iw, ic, iy, ix = np.where(self.max_args == 1)
        # ih2 & iw2 convert indices to input size
        ih2 = (ih * self.stride) + iy
        iw2 = (iw * self.stride) + ix
        # Use the indices to allocate the gradients correctly
        self.dinputs[im, ic, ih2, iw2] = grads[im, ic, ih, iw]

        return self.dinputs
