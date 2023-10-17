import numpy as np

from deepthink.layers.pooling.base_pooling import BasePooling
from deepthink.utils import get_strided_view_1D


class MaxPooling1D(BasePooling):
    """
    Max pooling operation for 1D data.

    Downsamples input data by taking the maximum value from each
    spatial pooling window. Used to reduce the spatial dimensions
    of the input data, which is normally expected to be temporal
    data.

    Input shape should be (batch, channels, seq_len)

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

    def __repr__(self):
        return 'MaxPooling1D'

    def initialize(self):
        """
        Initialize settings to prepare the layer for training
        """
        batches, channels, seq_len = self.input_shape
        self.batch_size = batches
        self.n_channels = channels
        self.seq_len = seq_len
        # Output size equation for 1D is: [(SEQ_LEN−K+2P)/S]+1
        self.output_size = ((seq_len - self.pool_size) / self.stride) + 1
        if int(self.output_size) != self.output_size:
            raise ValueError('Invalid dims. Output-size must be integer')

        self.output_size = int(self.output_size)
        self.output = np.zeros((batches, channels,
                               self.output_size),
                               dtype=self.dtype)
        # Create the shapes to use with "get_strided_view"
        self.forward_view_shape = (self.batch_size, self.output_size,
                                   self.n_channels, self.pool_size)

    def forward(self, inputs):
        """
        Perform one forward pass of MaxPooling operation.

        Parameters
        ----------
        inputs : np.array
            Input tensor with shape:
            (batch_size, channels, seq_len)

        Returns
        -------
        output : np.array
            Max-pooled output tensor with shape:
            (batch_size, channels, output_size)
        """
        # Reshape the input to a 4D tensor
        view = get_strided_view_1D(inputs,
                                   self.forward_view_shape,
                                   self.stride)
        # Find the maximum value in each pooling window
        self.output = np.max(view, axis=(3,), keepdims=True)
        # Create a mask of maximal values to use in backprop
        self.max_args = np.where(self.output == view, 1, 0)
        # Remove the last dimension of size 1
        self.output = np.squeeze(self.output, axis=(3,))
        # Transpose the output to match the expected shape
        self.output = self.output.transpose(0, 2, 1)
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

        # Use max_args to assign gradients to the correct indices
        batch, seq_len, channels, kernel = np.where(self.max_args == 1)
        # Convert indices to input size
        seq_in = (seq_len * self.stride) + kernel
        # Assign gradients to correct indices
        self.dinputs[batch, channels, seq_in] = grads[batch, channels, seq_len]

        return self.dinputs
