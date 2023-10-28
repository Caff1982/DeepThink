import numpy as np

from deepthink.layers.pooling.base_global_pooling import BaseGlobalPooling


class GlobalAveragePooling1D(BaseGlobalPooling):
    """
    Global Average Pooling 1D layer.

    Downsamples the input by taking the average value across the
    temporal dimension. This is useful for converting a sequence
    of feature vectors into a single feature vector.

    Parameters
    ----------
    axes : tuple, default=(-1,)
        The axis or axes along which the pooling is applied.
    keep_dims : bool, default=False
        Whether to keep the spatial dimensions or not. When False,
        the spatial dimensions are removed and the output shape is
        (batch_size, num_channels). When True, the spatial dimensions
        are retained and the output shape is (batch_size, num_channels,
        1).
    """
    def __init__(self, axes=(-1,), keep_dims=False, **kwargs):
        super().__init__(
            axes=axes,
            keep_dims=keep_dims,
            **kwargs)

    def backward(self, grads):
        """
        Perform backpropagation by computing the gradients.

        Parameters
        ----------
        grads : np.array, shape (batch_size, features)
            Gradients from the subsequent layer.

        Returns
        -------
        dinputs : np.array, shape (batch_size, features, seq_len)
            Gradients for the inputs.
        """
        # Reshape the gradients to (batch_size, num_channels, 1)
        # and divide by the number of elements in the feature map
        self.dinputs = grads[:, :, np.newaxis] / self.scaling_factor
        # Broadcast the gradients to the input shape using np.tile
        self.dinputs = np.tile(
            self.dinputs,
            (1, 1,
             self.spatial_size)
        )
        return self.dinputs
