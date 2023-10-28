import numpy as np

from deepthink.layers.pooling.base_pooling import BasePooling
from deepthink.utils import get_strided_view_3D


class MaxPooling3D(BasePooling):
    """
    Max pooling operation for 3D data.

    Downsamples input data by taking the maximum value from each
    spatial pooling window. When pooling window size and stride are
    both 2 (default values) the resulting array is halved along depth,
    height, width.

    Input shape should be (batch, channels, depth, height, width).

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
                                self.output_size, self.output_size,
                                self.output_size), dtype=self.dtype)
        # Adjust the forward view shape
        self.forward_view_shape = (self.batch_size, self.output_size,
                                   self.output_size, self.output_size,
                                   self.n_channels, self.pool_size,
                                   self.pool_size, self.pool_size)

    def forward(self, inputs):
        """
        Perform one forward pass of MaxPooling operation.

        Parameters
        ----------
        inputs : np.array
            Input tensor to perform forward pass on, with shape:
            (batch_size, channels, img_size_in, img_size_in, img_size_in)

        Returns
        -------
        output : np.array
            Max-pooled output tensor with shape:
            (batch_size, channels, img_size_out, img_size_out, img_size_out)
        """
        view = get_strided_view_3D(inputs,
                                   self.forward_view_shape,
                                   self.stride)
        self.output = np.max(view, axis=(5, 6, 7), keepdims=True)
        # Create a mask of the max values
        self.max_args = np.where(view == self.output, 1, 0)
        self.output = np.squeeze(self.output, axis=(5, 6, 7))
        self.output = np.transpose(self.output, (0, 4, 1, 2, 3))
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
        self.dinputs = np.zeros(self.input_shape)

        # Update the where operation for 3D
        im, id, ih, iw, ic, iz, iy, ix = np.where(self.max_args == 1)
        id2 = (id * self.stride) + iz
        ih2 = (ih * self.stride) + iy
        iw2 = (iw * self.stride) + ix

        self.dinputs[im, ic, id2, ih2, iw2] = grads[im, ic, id, ih, iw]

        return self.dinputs
