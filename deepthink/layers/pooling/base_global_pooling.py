import numpy as np

from deepthink.layers.layer import BaseLayer


class BaseGlobalPooling(BaseLayer):
    """Base class for global average pooling layers."""

    def __init__(self, axes, keep_dims=False, input_shape=None, **kwargs):
        super().__init__(
            input_shape=input_shape,
            **kwargs)
        self.axes = axes
        self.keep_dims = keep_dims

        # Create placeholders defined during 'set_output_shape_and_sizes'
        self.spatial_size = None  # spatial size of input
        self.scaling_factor = None  # Number of elements in ouput feature map

    def initialize(self):
        """
        Initialize settings to prepare the layer for training.

        Sets the output shape and initializes the output array.
        Also sets the spatial-size (i.e. sequence-length, image-size,
        etc) and scaling factor (i.e. number of elements in output)
        for use in backpropagation.
        """
        if self.keep_dims:
            shape = (self.input_shape[0], self.input_shape[1]) \
                     + (1,) * (len(self.input_shape) - 2)
        else:
            shape = (self.input_shape[0], self.input_shape[1])

        self.output = np.zeros(shape)

        self.spatial_size = self.input_shape[-1]
        self.scaling_factor = self.spatial_size ** len(self.axes)

    def forward(self, inputs):
        """
        Perform the forward pass on input tensor.

        Parameters
        ----------
        inputs : np.array
            Input tensor. The shape should be (batch, features,
            **spatial_dims). E.g. for 1D input (batch, features,
            spatial_size), 2D input (batch, features, height,
            width), etc.

        Returns
        -------
        output : np.array, shape (batch, features)
            Average-pooled output tensor.
        """
        self.input = inputs
        self.output = np.mean(inputs, axis=self.axes, keepdims=self.keep_dims)
        return self.output
