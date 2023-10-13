import numpy as np

from deepthink.utils import initialize_weights
from deepthink.layers.layer import BaseLayer


class Conv2D(BaseLayer):
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
        The size of step of the convolution window along both height
        and width dimensions.
    padding : int
        The amount of zero padding to add to the input image.

    References
    ----------
    - https://cs231n.github.io/convolutional-networks/#conv
    - https://www.youtube.com/watch?v=KuXjwB4LzSA&ab_channel=3Blue1Brown
    """
    def __init__(self, kernel_size, n_filters, stride=1,
                 padding=0, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.n_filters = n_filters
        self.stride = stride
        self.padding = padding

    def __str__(self):
        return 'Conv2D'

    def initialize(self):
        """
        Initialize settings to prepare the layer for training.
        """
        # If layer is not first layer use prev_layer to get input_shape
        if self.prev_layer and self.input_shape is None:
            self.input_shape = self.prev_layer.output.shape

        self.batch_size = self.input_shape[0]
        self.n_channels = self.input_shape[1]
        self.img_size = self.input_shape[2]
        # Output size equation is: [(W−K+2P)/S]+1
        self.output_size = ((self.img_size - self.kernel_size +
                            (2 * self.padding)) // self.stride) + 1
        # Ensure valid output width & depth size
        assert self.output_size % self.stride == 0
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
        self.backward_view_shape = (self.batch_size, self.img_size,
                                    self.img_size, self.n_filters,
                                    self.kernel_size, self.kernel_size)
        # Create padding constants to use in dilate_pad
        self.pad_L = self.kernel_size - self.padding - 1
        self.pad_R = self.kernel_size - 1
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

    def get_strided_view(self, arr, backward=False):
        """
        Return a view of an array using Numpy's as_strided slide-trick.

        Computationally efficient way to get all the kernel windows
        to be used in the convolution operation. Takes 4D tensor as
        input and outputs 6D tensor which can be used for the
        forward/backward pass operation.

        Parameters
        ----------
        arr : np.array
            The array/tensor to perform the operation on, should be 4D
            with shape (batch, depth, img-size, img-size)
        backward : bool,default=False
            Boolean used to state initialization the operation is
            forward or backward pass.

        Returns
        -------
        view : np.array
            The 6D view to be used in forward/backward pass

        References
        ----------
        - https://jessicastringham.net/2017/12/31/stride-tricks/
        - https://numpy.org/doc/stable/reference/generated/numpy.lib.stride_tricks.as_strided.html
        """
        if backward:
            # set stride to one to perform backprop
            stride = 1
            shape = self.backward_view_shape
        else:
            stride = self.stride
            shape = self.forward_view_shape
        # strides returns the byte-step for each dim in memory
        s0, s1, s2, s3 = arr.strides
        strides = (s0, stride * s2, stride * s3, s1, s2, s3)
        view = np.lib.stride_tricks.as_strided(
            arr, shape, strides=strides, writeable=True)
        return view

    def forward(self, X):
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

        Input shape should be (batch, depth-out, height, width).

        Parameters
        ----------
        X : np.array
            The input tensor to perform the convolution on.

        Returns
        -------
        output : np.array
            A tensor after the convolution has been applied

        References
        ----------
        -https://cs231n.github.io/convolutional-networks/#conv
        """
        # Add padding to height & width dimensions
        if self.padding > 0:
            X = np.pad(X,
                       pad_width=((0, 0), (0, 0),
                                  (self.padding, self.padding),
                                  (self.padding, self.padding)),
                       mode='constant')

        self.view = self.get_strided_view(X)
        # Reshape view to column vector
        X_col_dims = np.multiply.reduceat(self.view.shape, (0, 3))
        self.X_col = self.view.reshape(X_col_dims)
        # Reshape weights to column vector
        W_col = self.weights.reshape(self.n_filters, -1)
        # Perform dot product operation plus bias
        out = np.dot(W_col, self.X_col.T) + self.bias
        # Reshape and transpose back to output dimensions
        self.output = out.reshape(self.n_filters, self.batch_size,
                                  self.output_size, self.output_size)
        self.output = self.output.transpose(1, 0, 2, 3)
        return self.output

    def dilate_pad(self, arr):
        """
        Return gradients array with padding added.

        This method is used during backpropagation to add padding to
        the gradient array before performing matrix multiplication.
        Padding is added relative to stride.

        Parameters
        ----------
        arr : np.array
            The gradients array to perform the operation on. This
            should be the derivatives w.r.t inputs from the next layer.

        Returns
        -------
        dilated : np.array
            The array with padding added relative to stride, to be used
            in backpropagation matrix multiplication.
        """
        dilated = np.zeros(shape=self.dilate_pad_shape)
        dilated[:, :, ::self.stride, ::self.stride] = arr
        dilated = np.pad(dilated,
                         pad_width=((0, 0), (0, 0),
                                    (self.pad_L, self.pad_R),
                                    (self.pad_L, self.pad_R)),
                         mode='constant')

        return dilated

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
        padded_grads = self.dilate_pad(grads)
        # Rotate/transpose weights
        rot_weights = np.rot90(self.weights, 2, axes=(2, 3))
        grads_view = self.get_strided_view(padded_grads, True)
        dX = np.tensordot(grads_view, rot_weights, axes=([3, 4, 5],
                                                         [0, 2, 3]))
        self.dweights = dW
        # Reshape bias to column vector
        self.dbiases = dB.reshape(-1, 1)
        self.dinputs = dX.transpose(0, 3, 1, 2)
