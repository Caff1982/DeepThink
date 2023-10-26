from deepthink.layers.layer import BaseLayer


class BaseConv(BaseLayer):
    """
    Base class for convolution layers.

    Parameters
    ----------
    kernel_size : int
        The size of the convolution window.
    n_filters : int
        Number of filters in the convolution.
    stride : int, default=1
        The size of step of the convolution window along the sequence.
    padding_type : str, default='valid'
        The type of padding to use. Must be 'valid' or 'same'.
    input_shape : tuple, default=None
        The shape of the input to the layer. If None, the input shape is
        inferred from the previous layer.
    """
    def __init__(
        self,
        kernel_size,
        n_filters,
        stride=1,
        padding_type='valid',
        **kwargs,
    ):
        super().__init__(
            **kwargs
        )
        self.kernel_size = kernel_size
        self.n_filters = n_filters
        self.stride = stride
        self.padding_type = padding_type

        # Create placeholders for padding attributes
        self.padding_amount = 0
        self.padding = None  # Tuple of padding amounts
        self.dilate_padding = None  # Tuple of padding amounts

        # Create placeholders for attributes defined in initialize
        self.batch_size = None
        self.n_channels = None
        self.spatial_size = None  # Assumes square input for now
        self.output_size = None
        self.output = None  # Output of forward pass, np.array

        # Create placeholders for shapes used with get_strided_view
        self.forward_view_shape = None
        self.dilate_pad_shape = None
        self.backward_view_shape = None
        self.view = None

        # Placeholder arrays to store weights and biases
        self.weights = None
        self.bias = None
        # Placeholder arrays to store gradient estimates
        self.dweights = None
        self.dbiases = None
        self.dinputs = None
        # Placeholder arrays to store optimizer momentum values
        self.weight_momentum = None
        self.bias_momentum = None
        # Palceholder arrays to store optimizer gradient cache
        self.weight_grad_cache = None
        self.bias_grad_cache = None

    def _set_padding_and_output_size(self):
        """
        This method sets the padding and output size attributes based on
        the padding type and stride. It should be called in the initialize
        method of the child class, once the input shape is known.
        """
        # Set the padding to use in forward pass
        if self.padding_type == 'same':
            # Add padding to keep output size same as input
            self.padding_amount = ((self.spatial_size - 1) * self.stride
                                   + self.kernel_size - self.spatial_size) // 2
            self.padding = (self.padding_amount, self.padding_amount)
            self.output_size = self.spatial_size
        elif self.padding_type == 'valid':
            # Output size equation for is: [(spatial_size−K+2P)/S]+1
            self.output_size = ((self.spatial_size - self.kernel_size +
                                (2 * self.padding_amount)) // self.stride) + 1
        else:
            raise ValueError(
                "Invalid padding type. Must be 'valid' or 'same'."
            )

        if self.output_size % self.stride != 0:
            raise ValueError(
                'Invalid output dimensions. Try changing ',
                'kernel_size, stride or padding.')
