import numpy as np

from deepthink.layers.pooling.base_pooling import BasePooling


class GlobalAveragePooling3D(BasePooling):
    """
    Global Average Pooling 3D layer.

    Downsamples the input by taking the average value across the
    spatial dimensions. This is useful for converting a feature map
    into a single feature vector.

    Parameters
    ----------
    axes : tuple, default=(-3, -2, -1)
        The axes along which the pooling is applied.
    keep_dims : bool, default=False
        Whether to keep the spatial dimensions or not. When False,
        the spatial dimensions are removed and the output shape is
        (batch_size, num_channels). When True, the spatial dimensions
        are retained and the output shape is (batch_size, num_channels,
        1, 1, 1).
    """
    def __init__(self, axes=(-3, -2, -1), keep_dims=False, **kwargs):
        super().__init__(**kwargs)
        self.axes = axes
        self.keep_dims = keep_dims

    def __str__(self):
        return 'GlobalAveragePooling2D'

    def initialize(self):
        """
        Initialize the global average pooling 3D layer.
        """
        # The output shape will be (batch_size, num_channels)
        self.output = np.zeros((self.input_shape[0], self.input_shape[1]))

    def forward(self, inputs):
        """
        Perform the forward pass on input tensor.

        Parameters
        ----------
        inputs : np.array, shape (batch, features, height, width, depth)
            Input tensor.

        Returns
        -------
        output : np.array, shape (batch, features)
            Average-pooled output tensor.
        """
        self.input = inputs
        self.output = np.mean(inputs, axis=self.axes, keepdims=self.keep_dims)
        return self.output

    def backward(self, grads):
        """
        Perform backpropagation by computing the gradients.

        Parameters
        ----------
        grads : np.array, shape (batch_size, features)
            Gradients from the subsequent layer.

        Returns
        -------
        dinputs : np.array, shape (batch, features, height, width, depth)
            Gradients for the inputs.
        """
        height, width, depth = self.input_shape[-3:]
        # Reshape the gradients to shape (batch_size, num_channels, 1, 1)
        # and divide by the number of elements in the feature map
        self.dinputs = grads[:, :, np.newaxis, np.newaxis, np.newaxis] / (height * width * depth)
        # Broadcast the gradients to the input shape using np.tile
        self.dinputs = np.tile(self.dinputs, (1, 1, height, width, depth))
        return self.dinputs
