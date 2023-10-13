import numpy as np

from deepthink.utils import initialize_weights
from deepthink.layers.layer import BaseLayer


class MaxPooling(BaseLayer):
    """
    Max pooling operation for 2D data.

    Downsamples input data by taking the maximum value from each
    spatial pooling window. When pooling window size and stride are
    both 2 (default values) the resulting array is halved along
    height and width.

    Input shape should be (batch, channels, height, width).

    Parameters
    ----------
    size : int
        The size of the pooling window.
    stride : int
        The step size between each pooling window.
    padding : int
        The amount of zero padding to add to the input image
    """
    def __init__(self, size=2, stride=2, padding=0,
                 input_shape=None, **kwargs):
        super().__init__(**kwargs)
        self.size = size
        self.stride = stride
        self.padding = padding
        self.input_shape = input_shape

    def __repr__(self):
        return 'MaxPooling'

    def initialize(self):
        """
        Initialize settings to prepare the layer for training
        """
        if self.prev_layer is None and self.input_shape is None:
            raise ValueError('MaxPooling cannot be the first layer')

        if self.input_shape is None:
            self.input_shape = self.prev_layer.output.shape

        batches, channels, img_size, img_size = self.input_shape
        self.batch_size = batches
        self.n_channels = channels
        self.img_size = img_size
        # Output size equation is [(W−K+2P)/S]+1
        self.output_size = ((img_size - self.size +
                            (2 * self.padding)) / self.stride) + 1
        if int(self.output_size) != self.output_size:
            raise ValueError('Invalid dims. Output-size must be integer')
        self.output_size = int(self.output_size)
        self.output = np.zeros((batches, channels,
                               self.output_size, self.output_size),
                               dtype=self.dtype)
        # Create the shapes to use with "get_strided_view"
        self.forward_view_shape = (self.batch_size, self.output_size,
                                   self.output_size, self.n_channels,
                                   self.size, self.size)

    def get_strided_view(self, arr):
        """
        Return a view of an array using Numpy's as_strided
        slide-trick.

        Computationally efficient way to get all the kernel windows
        to be used in the convolution operation. Takes 4D tensor as
        input and outputs 6D tensor.

        Parameters
        ----------
        arr : np.array
            The array/tensor to perform the operation on, should
            be 4D with shape (batch, depth, img-size, img-size)

        Returns
        -------
        view : np.array
            The 6D view to be used in forward/backward pass
        """
        # strides returns the byte-step for each dim in memory
        s0, s1, s2, s3 = arr.strides
        strides = (s0, self.stride * s2, self.stride * s3, s1, s2, s3)
        view = np.lib.stride_tricks.as_strided(
            arr, self.forward_view_shape, strides=strides, writeable=True)
        return view

    def forward(self, X):
        """
        Perform one forward pass of MaxPooling operation.

        Parameters
        ----------
        X : np.array
            Input tensor with shape:
            (batch_size, channels, img_size_in, img_size_in)

        Returns
        -------
        output : np.array
            Max-pooled output tensor with shape:
            (batch_size, channels, img_size_out, img_size_out)
        """
        # Add padding to input array
        if self.padding:
            X = np.pad(X,
                       pad_width=((0, 0), (0, 0),
                                  (self.padding, self.padding),
                                  (self.padding, self.padding)),
                       mode='constant')

        view = self.get_strided_view(X)

        self.output = np.max(view, axis=(4, 5), keepdims=True)
        # Create a mask of maximal values to use in backprop
        self.max_args = np.where(self.output == view, 1, 0)
        self.output = np.squeeze(self.output, axis=(4, 5))
        self.output = self.output.transpose(0, 3, 1, 2)
        return self.output

    def backward(self, grads):
        """
        Perform one backward pass.

        Calculates partial derivatives w.r.t. inputs.
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
