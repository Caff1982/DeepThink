import numpy as np

from deepthink.layers.convolution.base_convolution import BaseConv
from deepthink.utils import initialize_weights, get_strided_view_2D, pad_2D


class Conv2D(BaseConv):
    """
    2D convolution layer.

    This layer convolves (or cross-correlates) a kernel with the input
    image to create a tensor of output feature maps.

    Input shape should be (batch_size, n_channels, img_size, img_size)
    and is required for the first layer.

    Parameters
    ----------
    kernel_size : int
        The width/height of the 2D convolution window.
        Currently only supports square kernels
    n_filters : int
        The dimensionality of the output space.
    stride : int, default=1
        The size of step of the convolution window along the sequence.
    padding_type : str, default='valid'
        The type of padding to use. Must be 'valid' or 'same'.
    input_shape : tuple, default=None
        The shape of the input to the layer. If None, the input shape is
        inferred from the previous layer.

    References
    ----------
    - https://cs231n.github.io/convolutional-networks/#conv
    - https://www.youtube.com/watch?v=KuXjwB4LzSA&ab_channel=3Blue1Brown
    """
    def __init__(
        self,
        kernel_size,
        n_filters,
        stride=1,
        padding_type='valid',
        input_shape=None,
        **kwargs,
    ):
        super().__init__(
            kernel_size=kernel_size,
            n_filters=n_filters,
            stride=stride,
            padding_type=padding_type,
            input_shape=input_shape,
            **kwargs
        )

    def initialize(self):
        """
        Initialize settings to prepare the layer for training.
        """
        self.batch_size = self.input_shape[0]
        self.n_channels = self.input_shape[1]
        self.spatial_size = self.input_shape[2]

        self._set_padding_and_output_size()

        # Create output array to store forward pass results
        self.output = np.zeros(
            (self.batch_size, self.n_filters,
             self.output_size, self.output_size)).astype(self.dtype)
        # Create the shapes to use with "get_strided_view"
        self.forward_view_shape = (self.batch_size, self.output_size,
                                   self.output_size, self.n_channels,
                                   self.kernel_size, self.kernel_size)
        self.dilate_pad_shape = (self.batch_size, self.n_filters,
                                 self.output_size * self.stride,
                                 self.output_size * self.stride)
        self.backward_view_shape = (self.batch_size, self.spatial_size,
                                    self.spatial_size, self.n_filters,
                                    self.kernel_size, self.kernel_size)

        # Initialize weights and bias
        kernel_shape = (self.n_filters, self.n_channels,
                        self.kernel_size, self.kernel_size)
        self.weights = initialize_weights(kernel_shape, self.weight_init,
                                          dtype=self.dtype)
        self.bias = np.zeros((self.n_filters, 1), dtype=self.dtype)
        # Initialize arrays to store optimizer momentum values
        self.weight_momentum = np.zeros(self.weights.shape, dtype=self.dtype)
        self.bias_momentum = np.zeros(self.bias.shape, dtype=self.dtype)
        # Initialize arrays to store optimizer gradient cache
        self.weight_grad_cache = np.zeros(self.weights.shape, dtype=self.dtype)
        self.bias_grad_cache = np.zeros(self.bias.shape, dtype=self.dtype)

    def forward(self, inputs):
        """
        Perform one forward pass of the convolution layer.

        Convolves an input tensor and outputs a tensor of shape
        (batch, depth-out, height-out, width-out).

        Numpy's as_strided function is used to create a view which
        is then reshaped to a column vector and dot product plus bias
        operation performed. This is a version of the image-to-column
        (im2col) algorithm which means that only one matrix
        multiplication is performed for each forward pass, improving
        computational efficiency.

        Input shape should be (batch, n_channels, height, width).

        Parameters
        ----------
        inputs : np.array
            The input tensor to perform the convolution on.

        Returns
        -------
        output : np.array
            A tensor after the convolution has been applied

        References
        ----------
        -https://cs231n.github.io/convolutional-networks/#conv
        """
        if self.padding:
            inputs = pad_2D(inputs, self.padding)

        self.view = get_strided_view_2D(inputs,
                                        self.forward_view_shape,
                                        self.stride)
        # Reshape view to column vector
        X_col_dims = np.multiply.reduceat(self.view.shape, (0, 3))
        X_col = self.view.reshape(X_col_dims)
        # Reshape weights to column vector
        W_col = self.weights.reshape(self.n_filters, -1)
        # Perform dot product operation plus bias
        out = np.dot(W_col, X_col.T) + self.bias
        # Reshape and transpose back to output dimensions
        self.output = out.reshape(self.n_filters,
                                  self.batch_size,
                                  self.output_size,
                                  self.output_size)
        self.output = self.output.transpose(1, 0, 2, 3)
        return self.output

    def backward(self, grads):
        """
        Perform one backward pass of the convolution layer.

        Partial derivatives are calculated w.r.t weights, biases
        and inputs.
        """
        # Get gradient w.r.t. bias
        dB = np.sum(grads, axis=(0, 2, 3))
        # Get gradient w.r.t weights
        dW = np.tensordot(self.view, grads, axes=([0, 1, 2],
                                                  [0, 2, 3]))
        dW = dW.transpose(3, 0, 1, 2)
        # Get gradients w.r.t inputs
        # Pad gradients to correct dims before matrix-multiply
        padded_grads = np.zeros(self.dilate_pad_shape)
        padded_grads[:, :, ::self.stride, ::self.stride] = grads
        padded_grads = pad_2D(padded_grads, self.dilate_padding)
        # Rotate/transpose weights
        rot_weights = np.rot90(self.weights, 2, axes=(2, 3))
        grads_view = get_strided_view_2D(padded_grads,
                                         self.backward_view_shape,
                                         1)
        dX = np.tensordot(grads_view, rot_weights, axes=([3, 4, 5],
                                                         [0, 2, 3]))
        self.dweights = dW
        # Reshape bias to column vector
        self.dbiases = dB.reshape(-1, 1)
        self.dinputs = dX.transpose(0, 3, 1, 2)
