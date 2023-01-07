import numpy as np

from deepthink.utils import initialize_weights


class BaseLayer:
    """
    Base layer class from which all layers inherit.

    A layer is an object that takes an input tensor and performs some
    operation on it, using the "forward" method. It also has a
    "backward" method for calculating derivatives as part of
    backpropagation. Calling 'initialization' is required to set
    prev_layer, next_layer, etc.

    Layers are designed to be chained together in a list which is an
    instance attribute of the Model class. One forward pass can then
    be completed by calling 'forward' on each subsequent layer and
    obtaining the output. Layers can also be used individually where
    required.

    Parameters
    ----------
    prev_layer : (any subclass of BaseLayer)
        The previous layer in the neural network. Added automatically
        during initialization.
    next_layer : (any subclass of BaseLayer)
        The next layer in the neural network. Added automatically
        during initialization.
    input_shape : tuple
        The shape of the input array. Computed during initialization.
    weight_init : str
        The type of initialization to use when creating the layer
        weights. See initialize_weights in utils for more info.
    dtype : type
        The numpy datatype to be used. Uses np.float32 by default,
        np.float64 is required for gradient checking.
    """
    def __init__(self, prev_layer=None, next_layer=None,
                 input_shape=None, weight_init='he_uniform',
                 dtype=np.float32, **kwargs):
        self.prev_layer = prev_layer
        self.next_layer = next_layer
        self.input_shape = input_shape
        self.weight_init = weight_init
        self.dtype = dtype

    def initialize(self):
        raise NotImplementedError(
            'All BaseLayer subclasses must implement initialize method'
        )

    def forward(self, X):
        raise NotImplementedError(
            'All BaseLayer subclasses must implement forward method'
        )

    def backward(self, grads):
        raise NotImplementedError(
            'All BaseLayer subclasses must implement backward method'
        )


class Dense(BaseLayer):
    """
    Fully-connected NN layer.

    During the forward pass it performs matrix multiplication between
    the inputs and weights with addition of bias as an offset.
    Backward pass computes derivatives for weights, bias and inputs.

    Parameters
    ----------
    n_outputs : int
        Dimensionality of output space, the number of neurons.
    n_inputs : int,default=None
        Input dimensions. Required if first layer, otherwise
        calculated during initialization
    """
    def __init__(self, n_outputs, n_inputs=None, **kwargs):
        super().__init__(**kwargs)
        self.n_outputs = n_outputs
        self.n_inputs = n_inputs

    def __str__(self):
        return 'Dense Layer'

    def initialize(self):
        """
        Initialize settings to prepare the layer for training
        """
        # If layer is not first layer use prev_layer to get input_shape
        if self.prev_layer and self.n_inputs is None:
            self.input_shape = self.prev_layer.output.shape
            self.n_inputs = self.input_shape[-1]
        # Initialize output array
        self.output = np.zeros((self.n_inputs, self.n_outputs),
                               dtype=self.dtype)
        # Initialize weights and biases
        self.weights = initialize_weights(self.output.shape,
                                          self.weight_init,
                                          self.dtype)
        self.bias = np.zeros((1, self.n_outputs),
                             dtype=self.dtype)
        # Initialize arrays to store optimizer momentum values
        self.weight_momentum = np.zeros(self.weights.shape,
                                        dtype=self.dtype)
        self.bias_momentum = np.zeros(self.bias.shape,
                                      dtype=self.dtype)
        # Initialize arrays to store optimizer gradient cache
        self.weight_grad_cache = np.zeros(self.weights.shape,
                                          dtype=self.dtype)
        self.bias_grad_cache = np.zeros(self.bias.shape,
                                        dtype=self.dtype)

    def forward(self, X):
        """
        Perform one forward pass on inputs X.
        """
        # Store input to be used in backpropagation
        self.input = X
        self.output = np.dot(X, self.weights) + self.bias
        return self.output

    def backward(self, grads):
        """
        Perform backpropagation by computing partial
        derivatives for weights, bias and inputs.
        """
        self.dweights = np.dot(self.input.T, grads)
        self.dbiases = np.sum(grads, axis=0, keepdims=True)
        self.dinputs = np.dot(grads, self.weights.T)


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


class MaxPooling(BaseLayer):
    """
    Max pooling operation for 2D data.

    Downsamples input data by taking the maximum value from each
    spatial pooling window. When pooling window size and stride are
    both 2 (default values) the resulting array is halved along height
    and width.

    Parameters
    ----------
    size : int
        The size of the pooling window.
    stride : int
        The step size between each pooling window.
    padding : int
        The amount of zero padding to add to the input image
    """
    def __init__(self, size=2, stride=2, padding=0, **kwargs):
        super().__init__(**kwargs)
        self.size = size
        self.stride = stride
        self.padding = padding

    def __repr__(self):
        return 'MaxPooling'

    def initialize(self):
        """
        Initialize settings to prepare the layer for training
        """
        # If layer is not first layer use prev_layer to get input_shape
        if self.input_shape is None:
            self.input_shape = self.prev_layer.output.shape

        batches, channels, img_size, img_size = self.input_shape
        self.batch_size = batches
        self.n_channels = channels
        self.img_size = img_size
        # Output size equation is [(W−K+2P)/S]+1
        self.output_size = ((img_size - self.size +
                            (2 * self.padding)) / self.stride) + 1
        if int(self.output_size) != self.output_size:
            raise Exception('Invalid dims. Output-size must be integer')
        self.output_size = int(self.output_size)
        self.output = np.zeros((batches, channels,
                               self.output_size, self.output_size),
                               dtype=self.dtype)
        # Create the shapes to use with "get_strided_view"
        self.forward_view_shape = (self.batch_size, self.output_size,
                                   self.output_size, self.n_channels,
                                   self.size, self.size)

    def get_strided_view(self, arr):
        """
        Return a view of an array using Numpy's as_strided slide-trick.

        Computationally efficient way to get all the kernel windows
        to be used in the convolution operation. Takes 4D tensor as
        input and outputs 6D tensor.

        Parameters
        ----------
        arr : np.array
            The array/tensor to perform the operation on, should
            be 4D with shape (batch, depth, img-size, img-size)

        Returns
        -------
        view : np.array
            The 6D view to be used in forward/backward pass
        """
        # strides returns the byte-step for each dim in memory
        s0, s1, s2, s3 = arr.strides
        strides = (s0, self.stride * s2, self.stride * s3, s1, s2, s3)
        view = np.lib.stride_tricks.as_strided(
            arr, self.forward_view_shape, strides=strides, writeable=True)
        return view

    def forward(self, X):
        """
        Perform one forward pass of MaxPooling operation
        """
        # Add padding to input array
        if self.padding:
            X = np.pad(X,
                       pad_width=((0, 0), (0, 0),
                                  (self.padding, self.padding),
                                  (self.padding, self.padding)),
                       mode='constant')

        view = self.get_strided_view(X)

        self.output = np.max(view, axis=(4, 5), keepdims=True)
        # Create a mask of maximal values to use in backprop
        self.max_args = np.where(self.output == view, 1, 0)
        self.output = np.squeeze(self.output, axis=(4, 5))
        self.output = self.output.transpose(0, 3, 1, 2)
        return self.output

    def backward(self, grads):
        """
        Perform one backward pass.

        Calculates partial derivatives w.r.t. inputs.
        """
        # Initialize empty array
        self.dinputs = np.zeros(self.input_shape)
        # Use max_args mask to get the maximal indices
        im, ih, iw, ic, iy, ix = np.where(self.max_args == 1)
        # ih2 & iw2 convert indices to input size
        ih2 = (ih * self.stride) + iy
        iw2 = (iw * self.stride) + ix
        # Use the indices to allocate the gradients correctly
        self.dinputs[im, ic, ih2, iw2] = grads[im, ic, ih, iw].flatten()

        return self.dinputs


class Flatten(BaseLayer):
    """
    Flattens the input array, batch-size is the only dimension
    preserved. Used to reshape output from conv/pooling layers to
    input for  a full-connected Dense layer.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __repr__(self):
        return 'Flatten'

    def initialize(self):
        """
        Initialize settings to prepare the layer for training
        """
        self.input_shape = self.prev_layer.output.shape
        self.output = np.zeros(
            self.input_shape,
            dtype=self.dtype).reshape((self.input_shape[0], -1))

    def forward(self, X):
        """
        Perform one forward pass by flattening input array
        """
        self.output = X.ravel().reshape(self.output.shape)
        return self.output

    def backward(self, grads):
        """
        Perform backward pass.

        Gradients are reshaped from 1D to their
        input dimensions.
        """
        self.dinputs = grads.reshape(self.input_shape)
