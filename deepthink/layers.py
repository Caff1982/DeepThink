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
        The shape of the input array. This is a required argument
        for the first layer, otherwise computed during
        initialization. Should have shape (N, D) or (N, D, H, W).
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
    n_neurons : int
        The number of neurons, dimensionality of output space.
    input_shape : tuple,default=None
        Input shape. Required if first layer, otherwise calculated
        during initialization. Should have shape (N, D).
    """
    def __init__(self, n_neurons, input_shape=None, **kwargs):
        super().__init__(**kwargs)
        self.n_neurons = n_neurons
        self.input_shape = input_shape

    def __str__(self):
        return f'Dense({self.n_neurons})'

    def initialize(self):
        """
        Initialize settings to prepare the layer for training
        """
        # If layer is not first layer use prev_layer to get input_shape
        if self.prev_layer:
            self.input_shape = self.prev_layer.output.shape
        elif self.prev_layer is None and self.input_shape is None:
            raise Exception(
                'input_shape is a required argument for the first layer.')
        # Initialize output array, shape: (batch-size, n_neurons)
        self.output = np.zeros((self.input_shape[0], self.n_neurons),
                               dtype=self.dtype)
        # Initialize weights and biases.
        # Weights have shape: (input-dimensions, output-dimensions)
        weights_shape = (self.input_shape[-1], self.n_neurons)
        self.weights = initialize_weights(weights_shape,
                                          self.weight_init,
                                          self.dtype)
        self.bias = np.zeros((1, self.n_neurons),
                             dtype=self.dtype)
        # Initialize arrays to store optimizer momentum values
        self.weight_momentum = np.zeros(weights_shape,
                                        dtype=self.dtype)
        self.bias_momentum = np.zeros(self.bias.shape,
                                      dtype=self.dtype)
        # Initialize arrays to store optimizer gradient cache
        self.weight_grad_cache = np.zeros(weights_shape,
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


class Conv1D(BaseLayer):
    """
    1D convolution layer.

    This layer convolves (or cross-correlates) a kernel with the input
    sequence to create a tensor of output feature maps.

    Input shape should be (batch_size, n_channels, sequence_length)
    and is required for the first layer.

    Parameters
    ----------
    kernel_size : int
        The width of the 1D convolution window.
    n_filters : int
        The dimensionality of the output space.
    stride : int, default=1
        The size of step of the convolution window along the sequence.
    padding : int
        The amount of zero padding to add to the input sequence.
    
    References
    ----------
    - https://cs231n.github.io/convolutional-networks/#conv
    - https://www.youtube.com/watch?v=KuXjwB4LzSA&ab_channel=3Blue1Brown
    """
    def __init__(self, kernel_size, n_filters, stride=1, padding=0, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.n_filters = n_filters
        self.stride = stride
        self.padding = padding

    def __str__(self):
        return 'Conv1D'

    def initialize(self):
        """
        Initialize settings to prepare the layer for training.
        """
        # If layer is not first layer use prev_layer to get input_shape
        if self.prev_layer and self.input_shape is None:
            self.input_shape = self.prev_layer.output.shape

        self.batch_size = self.input_shape[0]
        self.n_channels = self.input_shape[1]
        self.sequence_length = self.input_shape[2]
        # Output size equation for 1D is: [(SEQ_LEN−K+2P)/S]+1
        self.output_length = ((self.sequence_length - self.kernel_size +
                              (2 * self.padding)) // self.stride) + 1
        assert self.output_length % self.stride == 0
        self.output = np.zeros(
            (self.batch_size,
             self.n_filters,
             self.output_length)).astype(self.dtype)
        
        # Create the shapes to use with "get_strided_view"
        self.forward_view_shape = (self.batch_size, self.output_length,
                                   self.n_channels, self.kernel_size)
        self.dilate_pad_shape = (self.batch_size, self.n_filters,
                                 self.output_length * self.stride)
        self.backward_view_shape = (self.batch_size, self.sequence_length,
                                    self.n_filters, self.kernel_size)
        # Create padding constants to use in dilate_pad
        self.pad_L = self.kernel_size - self.padding - 1
        self.pad_R = self.kernel_size - 1
        # Initialize weights and bias
        self.kernel_shape = (self.n_filters, self.n_channels, self.kernel_size)
        self.weights = initialize_weights(self.kernel_shape,
                                          self.weight_init,
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
        to be used in the convolution operation. Takes 3D tensor as
        input and outputs 4D tensor which can be used for the
        forward/backward pass operation.

        Parameters
        ----------
        arr : np.array
            The array/tensor to perform the operation on, should be 3D
            with shape (batch, depth, sequence-length)
        backward : bool,default=False
            Boolean used to state initialization the operation is
            forward or backward pass.
        
        Returns
        -------
        view : np.array
            The 4D view to be used in forward/backward pass
        
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
        s0, s1, s2 = arr.strides
        strides = (s0, stride * s2, s1, s2)
        view = np.lib.stride_tricks.as_strided(
            arr, shape, strides=strides, writeable=True)
        return view

    def forward(self, X):
        """
        Perform one forward pass of the convolution layer.

        Convolves an input tensor and outputs a tensor of shape
        (batch, depth-out, sequence-out).

        Numpy's as_strided function is used to create a view which
        is then reshaped to a column vector and dot product plus bias
        operation performed. This is a version of the image-to-column
        (im2col) algorithm which means that only one matrix
        multiplication is performed for each forward pass, improving
        computational efficiency.

        Input shape should be (batch, depth-out, sequence).

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
        # Add padding to sequence dimension
        if self.padding > 0:
            X = np.pad(X,
                       pad_width=((0, 0), (0, 0),
                                  (self.padding,
                                   self.padding)),
                        mode='constant')

        self.view = self.get_strided_view(X)
        # Reshape view to column vector
        X_col_dims = np.multiply.reduceat(self.view.shape, (0, 2))
        self.X_col = self.view.reshape(X_col_dims)
        # Reshape weights to column vector
        W_col = self.weights.reshape(self.n_filters, -1)
        # Perform dot product operation plus bias
        out = np.dot(W_col, self.X_col.T) + self.bias
        # Reshape and transpose back to output dimensions
        self.output = out.reshape(self.n_filters,
                                  self.batch_size,
                                  self.output_length)
        self.output = self.output.transpose(1, 0, 2)
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
        dilated[:, :, ::self.stride] = arr
        dilated = np.pad(dilated,
                         pad_width=((0, 0), (0, 0),
                                    (self.pad_L, self.pad_R)),
                         mode='constant')

        return dilated

    def backward(self, grads):
        """
        Perform one backward pass of the convolution layer.

        Partial derivatives are calculated w.r.t weights, biases
        and inputs.
        """
        # Get the gradient w.r.t. bias
        dB = np.sum(grads, axis=(0, 2))
        # Get the gradient w.r.t. weights
        dW = np.tensordot(self.view, grads, axes=([0, 1], [0, 2]))
        dW = dW.transpose(2, 0, 1)
        # Get the gradient w.r.t. inputs
        # Pad gradients to correct dims before matrix-multiply
        padded_grads = self.dilate_pad(grads)
        # Rotate/transpose weights
        rot_weights = self.weights[:, :, ::-1]
        grads_view = self.get_strided_view(padded_grads, True)
        dX = np.tensordot(grads_view, rot_weights, axes=([2, 3],
                                                         [0, 2]))
        self.dweights = dW
        # Reshape bias to column vector
        self.dbiases = dB.reshape(-1, 1)
        self.dinputs = dX.transpose(0, 2, 1)


class EmbeddingLayer(BaseLayer):
    """
    Embedding layer for mapping discrete entities to dense vectors.

    Parameters
    ----------
    vocab_size : int
        The number of unique entities in the vocabulary.
    emb_dims : int
        The number of dimensions to map each entity to.
    input_shape : tuple
        The shape of the input tensor. E.g. (batch, sequence_length).
    weight_init : str,default='uniform'
        The weight initialization strategy to use for the weights.
    """
    def __init__(self, vocab_size, emb_dims, input_shape,
                 weight_init='uniform', **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.emb_dims = emb_dims
        self.weight_init = weight_init
        self.input_shape = input_shape

    def __str__(self):
        return 'Embedding Layer'

    def initialize(self):
        """
        Initialize the embedding layer.
        """
        self.seq_len = self.input_shape[-1]
        weights_shape = (self.vocab_size, self.emb_dims)
        self.weights = initialize_weights(weights_shape,
                                          self.weight_init,
                                          self.dtype)
        self.output = np.zeros((self.input_shape[0],
                                self.emb_dims,
                                self.seq_len))
        # Initialize arrays to store optimizer momentum values
        self.weight_momentum = np.zeros(weights_shape, dtype=self.dtype)
        # Initialize arrays to store optimizer gradient cache
        self.weight_grad_cache = np.zeros(weights_shape, dtype=self.dtype)
        # Placeholder for input grads, not used in Embedding layer
        self.dinputs = None

    def forward(self, X):
        """
        Perform the forward pass on inputs X.

        Parameters
        ----------
        X : array-like, shape (batch_size, input_length)
            Input sequence of word indices.

        Returns
        -------
        output : array-like, shape (batch_size, input_length, output_dim)
            Embedding vectors for the input sequence.
        """
        self.input = X
        self.output = self.weights[self.input]
        # Transpose to ensure channels first
        return self.output.transpose(0, 2, 1)

    def backward(self, grads):
        """
        Perform backpropagation by computing the gradients.

        Note that the gradients w.r.t. the inputs are not calculated
        as this layer is typically used as the first layer in a network.

        Parameters
        ----------
        grads : array-like
            Gradients from the subsequent layer.
        """
        self.dweights = np.zeros_like(self.weights)

        np.add.at(self.dweights, self.input, grads.transpose(0, 2, 1))


class MaxPooling(BaseLayer):
    """
    Max pooling operation for 2D data.

    Downsamples input data by taking the maximum value from each
    spatial pooling window. When pooling window size and stride are
    both 2 (default values) the resulting array is halved along
    height and width.

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
        if self.prev_layer is None and self.input_shape is None:
            raise ValueError('MaxPooling cannot be the first layer')

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
        Return a view of an array using Numpy's as_strided
        slide-trick.

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
        self.dinputs[im, ic, ih2, iw2] = grads[im, ic, ih, iw]

        return self.dinputs


class GlobalAveragePooling1D(BaseLayer):
    """
    Global Average Pooling layer.

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
        if self.prev_layer is None and self.input_shape is None:
            raise ValueError('GlobalAveragePooling1D cannot be the first layer')

        if self.input_shape is None:
            self.input_shape = self.prev_layer.output.shape

        self.output = np.zeros((self.input_shape[0], self.input_shape[1]))

    def forward(self, X):
        """
        Perform the forward pass on inputs X.

        Parameters
        ----------
        X : array-like, shape (batch_size, features, seq_len)
            Input tensor.

        Returns
        -------
        output : array-like, shape (batch_size, features)
            Average-pooled output tensor.
        """
        self.input = X
        self.output = np.mean(X, axis=self.axis)
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


class GlobalAveragePooling2D(BaseLayer):
    """
    Global Average Pooling 2D layer.

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
        if self.prev_layer is None and self.input_shape is None:
            raise ValueError('GlobalAveragePooling2D cannot be the first layer')

        if self.input_shape is None:
            self.input_shape = self.prev_layer.output.shape

        # The output shape will be (batch_size, num_channels)
        self.output = np.zeros((self.input_shape[0], self.input_shape[1]))

    def forward(self, X):
        """
        Perform the forward pass on inputs X.

        Parameters
        ----------
        X : array-like, shape (batch_size, features, height, width)
            Input tensor.

        Returns
        -------
        output : array-like, shape (batch_size, features)
            Average-pooled output tensor.
        """
        self.input = X
        self.output = np.mean(X, axis=self.axes)
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
        # Broadcast the gradients to the input shape
        self.dinputs = self.dinputs.repeat(height, axis=-2).repeat(width, axis=-1)
        return self.dinputs


class Flatten(BaseLayer):
    """
    Flattens the input array, batch-size is the only dimension
    preserved. Used to reshape output from conv/pooling layers to
    be used as input for a Dense layer.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __repr__(self):
        return 'Flatten'

    def initialize(self):
        """
        Initialize settings to prepare the layer for training
        """
        if self.prev_layer is None and self.input_shape is None:
            raise ValueError('Flatten cannot be the first layer')

        if self.input_shape is None:
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


class Dropout(BaseLayer):
    """
    Applies dropout to the input during the forward pass.

    The Dropout layer creates a mask to randomly set input units
    to zero with probabilty 'proba'. This mask is stored as an
    attribute which is used to propogate the gradients during
    backpropogation.

    Parameters
    ----------
    proba: float
        The dropout probability of neurons to keep. The dropout mask
        is created by sampling from a Bernoulli distribution with
        this probability.

    References
    ----------
    - https://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf
    """
    def __init__(self, proba, **kwargs):
        super().__init__(**kwargs)
        self.proba = proba

    def __repr__(self):
        return f'Dropout-{self.proba}'

    def initialize(self):
        """
        Initialize settings to prepare the layer for training
        """
        if self.prev_layer is None and self.input_shape is None:
            raise ValueError('Dropout cannot be the first layer')

        if self.input_shape is None:
            self.input_shape = self.prev_layer.output.shape

        self.output = np.zeros(self.input_shape, dtype=self.dtype)

    def forward(self, X, training=True):
        """
        Apply dropout to the input.

        Parameters
        ----------
        X: numpy.ndarray
            The input tensor to which dropout is applied.
        training : bool,default=True
            If set to True, applies dropout to the input during
            training mode. If set to False, no dropout is applied
            during inference mode.

        Returns
        -------
        output : numpy.ndarray
            The input with dropout applied.
        """
        if training:
            self.mask = np.random.binomial(1, self.proba,
                                           size=self.input_shape) / self.proba
            self.output = X * self.mask
        else:
            self.output = X.copy()
        return self.output

    def backward(self, grads):
        """
        Propagate gradients through the dropout mask during
        backpropagation.

        Parameters
        ----------
        grads: numpy.ndarray
            The gradient of the loss with respect to the output of
            the dropout layer.

        Returns
        -------
        output : numpy.ndarray
            The gradient of the loss with respect to the input of
            the dropout layer.
        """
        self.dinputs = grads * self.mask


class BatchNorm(BaseLayer):
    """
    Batch Normalization layer for neural networks.

    Batch normalization normalizes the input values by subtracting
    the mean and dividing by the standard deviation, so that the
    output has zero mean and unit variance. During training, the
    batch mean and variance are used. However, for testing and
    inference, a running mean and variance are kept and used.

    This layer can be applied to both fully-connected and
    convolutional layers. The total number of elements are stored
    in the 'Nt' attribute so that operations can be performed over
    the channel dimension, regardless of input size.

    Parameters
    ----------
    epsilon : float, default=1e-5
        A small constant added to the variance to avoid division by
        zero errors.
    mom : float, default=0.9
        The momentum for the running mean and variance.
    kwargs:
        Additional keyword arguments passed to the BaseLayer.

    References
    ----------
    - https://arxiv.org/pdf/1502.03167.pdf
    """
    def __init__(self,  epsilon=1e-5, mom=0.9, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.mom = mom

    def __repr__(self):
        return 'Batch Normalization'

    def initialize(self):
        """
        Initialize settings to prepare the layer for training.

        - self.weights and self.bias represent the gamma and beta
          parameters in the Batch Normalization paper respectively.
        - self.dweights and self.dbiases are used to store the
          gradients of the loss w.r.t. the gamma and beta parameters
          respectively.
        - self.running_mean and self.running_var store the running
          mean and variance which are used during inference.
        """
        if self.prev_layer is None and self.input_shape is None:
            raise ValueError('BatchNorm cannot be the first layer')

        if self.input_shape is None:
            self.input_shape = self.prev_layer.output.shape

        # Check input-dims to set params for 2 or 4 input dims
        if len(self.input_shape) == 4:
            self.N, self.C, self.H, self.W = self.input_shape
            shape = (1, self.C, 1, 1)
            self.axes = (0, 2, 3)
            self.keepdims = True
            self.Nt = self.N * self.H * self.W
        else:
            self.N, self.C = self.input_shape
            shape = (self.C,)
            self.axes = 0
            self.keepdims = False
            self.Nt = self.N

        # weights & bias represent gamma and beta in paper
        self.weights = np.ones(shape, dtype=self.dtype)
        self.bias = np.zeros(shape, dtype=self.dtype)
        self.dweights = np.zeros(shape, dtype=self.dtype)
        self.dbiases = np.zeros(shape, dtype=self.dtype)
        # Running mean & var initialized during first forward pass
        self.running_mean = None
        self.running_var = None

        # Initialize arrays to store optimizer momentum values
        self.weight_momentum = np.zeros(shape, dtype=self.dtype)
        self.bias_momentum = np.zeros(shape, dtype=self.dtype)
        # Initialize arrays to store optimizer gradient cache
        self.weight_grad_cache = np.zeros(shape, dtype=self.dtype)
        self.bias_grad_cache = np.zeros(shape, dtype=self.dtype)

        self.output = np.zeros(self.input_shape, dtype=self.dtype)

    def forward(self, X, training=True):
        """
        Perform the forward pass of the batch normalization layer.

        During training, the mean and variance of the current batch
        are used to normalize the input, and the running mean and
        variance are updated with a momentum term to keep track of
        the historical statistics of the data. During inference, the
        running mean and variance are used instead of the current
        batch mean and variance to normalize the input and ensure
        stable output across different batches.

        Parameters
        ----------
        X : numpy.ndarray
            The input tensor to apply batch normalization to.
        training : bool, default=True
            Flag indicating whether the layer is in training mode
            or not.

        Returns
        -------
        numpy.ndarray
            The normalized output tensor.
        """
        # Running mean & var set to batch mean/var on first pass
        if self.running_mean is None:
            self.running_mean = np.mean(X, axis=self.axes,
                                        keepdims=self.keepdims)
            self.running_var = np.var(X, axis=self.axes,
                                      keepdims=self.keepdims)

        if training:
            # If training use batch mean and batch var
            mean = np.mean(X, axis=self.axes, keepdims=self.keepdims)
            self.var = np.var(X, axis=self.axes, keepdims=self.keepdims)
            # Update running mean/var
            self.running_mean = (1 - self.mom) * mean \
                                 + self.mom * self.running_mean
            self.running_var = (1 - self.mom) * self.var \
                                + self.mom * self.running_var
        else:
            # If test mode use running mean and var
            mean = self.running_mean
            self.var = self.running_var
        # Perform forward pass and store attributes for backpass
        self.X_mu = X - mean
        self.X_norm = self.X_mu / np.sqrt(self.var + self.epsilon)
        self.output = self.weights * self.X_norm + self.bias
        return self.output

    def backward(self, grads):
        """
        Perform the backward pass of the batch normalization layer.

        During the backward pass, the gradient of the loss with
        respect to the input of this layer is computed using the
        following steps:

        1. The gradients of the loss with respect to the bias and
           weights are computed by summing the gradients along the
           appropriate axes, as specified by the self.axes and
           self.keepdims attributes. These gradients are stored as
           self.dbiases and self.dweights respectively.
        2. The total number of elements in the input, Nt, is
           calculated based on the input shape. If the input has 4
           dimensions (i.e. it's a convolutional layer), Nt is
           calculated as N * H * W. If the input has 2 dimensions
           (i.e. it's a fully-connected layer), Nt is simply N.
        3. The gradients of the loss with respect to the inputs are
           computed by multiplying Nt by the gradients, subtracting
           the sum of the gradients along the appropriate axes, and
           subtracting the product of X_mu and the sum of the
           gradients multiplied by X_mu, all scaled by the square
           root of the variance plus epsilon. The final result is
           also multiplied by the weights. This is stored as
           self.dinputs.

        Parameters
        ----------
        grads : numpy.ndarray
            The gradient of the loss with respect to the output of
            this layer.

        Returns
        -------
        numpy.ndarray
            The gradient of the loss with respect to the input of
            this layer.

        References
        ----------
        - https://github.com/cthorey/CS231/blob/master/assignment2/cs231n/layers.py
        """
        # Get derivatives w.r.t bias and weights
        self.dbiases = np.sum(grads, axis=self.axes, keepdims=self.keepdims)
        self.dweights = np.sum(self.X_norm * grads, axis=self.axes,
                               keepdims=self.keepdims)

        self.dinputs = ((1.0 / self.Nt) * self.weights
                        * np.sqrt(self.var + self.epsilon)
                        * (self.Nt * grads - np.sum(grads, axis=self.axes,
                                                    keepdims=self.keepdims)
                        - self.X_mu * (self.var + self.epsilon)**-1
                        * np.sum(grads * self.X_mu, axis=self.axes,
                                 keepdims=self.keepdims)))
        return self.dinputs
