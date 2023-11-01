import numpy as np

from deepthink.layers.layer import BaseLayer


class Upsample1D(BaseLayer):
    """
    Upsample layer for 1D inputs.

    Images are resized using nearest neighbor interpolation.

    Parameters
    ----------
    scale_factor : int, default=2
        Upsampling factor. The output size is computed as
        `scale_factor * input_size`.
    """

    def __init__(self, scale_factor=2, **kwargs):
        super().__init__(**kwargs)
        self.scale_factor = scale_factor

    def initialize(self):
        """Initialize output shape."""
        upsample_size = self.input_shape[-1] * self.scale_factor
        self.output = np.zeros((
            self.input_shape[0],
            self.input_shape[1],
            upsample_size),
            dtype=self.dtype)

    def forward(self, inputs):
        """
        Perform forward pass.

        Parameters
        ----------
        inputs : np.array
            Input data of shape (batch_size, channels, seq-length).

        Returns
        -------
        np.array
            Output data of shape (batch_size, channels,
            seq-length * scale_factor).

        """
        self.output = np.repeat(
            inputs,
            self.scale_factor,
            axis=2
        )
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
        self.dinputs = np.zeros(self.input_shape, dtype=self.dtype)
        self.dinputs = grads[:, :, ::self.scale_factor]
        return self.dinputs
