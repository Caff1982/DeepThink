from deepthink.layers.layer import BaseLayer


class BasePooling(BaseLayer):
    """Base class for max pooling layers."""

    def __init__(
        self,
        pool_size=2,
        stride=2,
        input_shape=None,
        **kwargs
    ):
        super().__init__(
            input_shape=input_shape,
            **kwargs)
        self.pool_size = pool_size
        self.stride = stride

        # Create placeholders defined during 'set_output_size'
        self.batch_size = None
        self.n_channels = None
        self.spatial_size = None  # spatial size of input
        self.output_size = None  # spatial size of output
        self.forward_view_shape = None  # shape for get_strided_view
        self.max_args = None  # indices of max values for backprop

    def set_output_size(self):
        """
        Set the output size of the layer.

        Calculates the output spatial size based on the input shape and
        pooling parameters. Also sets the spatial size in (i.e.
        image-size, sequence length, etc), number of channels, and batch
        size.
        """
        batches, channels, spatial_size = self.input_shape[:3]
        self.batch_size = batches
        self.n_channels = channels
        self.spatial_size = spatial_size

        self.output_size = ((spatial_size - self.pool_size) // self.stride) + 1
