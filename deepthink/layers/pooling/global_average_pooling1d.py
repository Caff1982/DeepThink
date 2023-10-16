import numpy as np

from deepthink.layers.pooling.base_pooling import BasePooling


class GlobalAveragePooling1D(BasePooling):
    """
    Global Average Pooling layer.

    Downsamples the input by taking the average value across the
    temporal dimension. This is useful for converting a sequence
    of feature vectors into a single feature vector.

    Parameters
    ----------
    axis : int or tuple, default=-1
        The axis or axes along which the pooling is applied.
    """
    def __init__(self, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def __str__(self):
        return 'GlobalAveragePooling1D'

    def initialize(self):
        """
        Initialize the global average pooling layer.
        """
        # The output shape will be (batch_size, num_channels)
        self.output = np.zeros((self.input_shape[0], self.input_shape[1]))

    def forward(self, inputs):
        """
        Perform the forward pass on input tensor.

        Parameters
        ----------
        inputs : np.array, shape (batch_size, features, seq_len)
            Input tensor.

        Returns
        -------
        output : np.array, shape (batch_size, features)
            Average-pooled output tensor.
        """
        self.input = inputs
        self.output = np.mean(inputs, axis=self.axis)
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
        dinputs : array-like, shape (batch_size, features, seq_len)
            Gradients for the inputs.
        """
        seq_len = self.input_shape[-1]
        self.dinputs = np.ones(grads.shape) * grads / seq_len
        # Broadcast the gradients to the input shape
        self.dinputs = self.dinputs[..., np.newaxis].repeat(seq_len, axis=-1)
        return self.dinputs
