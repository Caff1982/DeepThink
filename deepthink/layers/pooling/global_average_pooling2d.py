import numpy as np

from deepthink.layers.pooling.base_pooling import BasePooling


class GlobalAveragePooling2D(BasePooling):
    """
    Global Average Pooling 2D layer.

    Downsamples the input by taking the average value across the
    spatial dimensions. This is useful for converting a feature map
    into a single feature vector.

    Parameters
    ----------
    axes : tuple, default=(-2, -1)
        The axes along which the pooling is applied.
    """
    def __init__(self, axes=(-2, -1), **kwargs):
        super().__init__(**kwargs)
        self.axes = axes

    def __str__(self):
        return 'GlobalAveragePooling2D'

    def initialize(self):
        """
        Initialize the global average pooling 2D layer.
        """
        # The output shape will be (batch_size, num_channels)
        self.output = np.zeros((self.input_shape[0], self.input_shape[1]))

    def forward(self, inputs):
        """
        Perform the forward pass on input tensor.

        Parameters
        ----------
        X : np.array, shape (batch_size, features, height, width)
            Input tensor.

        Returns
        -------
        output : np.array, shape (batch_size, features)
            Average-pooled output tensor.
        """
        self.input = inputs
        self.output = np.mean(inputs, axis=self.axes)
        return self.output

    def backward(self, grads):
        """
        Perform backpropagation by computing the gradients.

        Parameters
        ----------
        grads : array-like, shape (batch_size, features)
            Gradients from the subsequent layer.

        Returns
        -------
        dinputs : array-like, shape (batch_size, features, height, width)
            Gradients for the inputs.
        """
        height, width = self.input_shape[-2], self.input_shape[-1]
        # Reshape the gradients to shape (batch_size, num_channels, 1, 1)
        # and divide by the number of elements in the feature map
        self.dinputs = grads[:, :, np.newaxis, np.newaxis] / (height * width)
        # Broadcast the gradients to the input shape using np.tile
        self.dinputs = np.tile(self.dinputs, (1, 1, height, width))
        return self.dinputs
