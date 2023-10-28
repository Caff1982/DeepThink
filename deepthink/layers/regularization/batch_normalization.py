import numpy as np

from deepthink.utils import initialize_weights
from deepthink.layers.layer import BaseLayer


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
    def __init__(self,  epsilon=1e-5, mom=0.9, input_shape=None, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.mom = mom
        self.input_shape = input_shape

    def __repr__(self):
        return 'BatchNormalization'

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
        - https://arxiv.org/pdf/1502.03167.pdf
        """
        # Get derivatives w.r.t bias and weights
        self.dbiases = np.sum(grads, axis=self.axes, keepdims=self.keepdims)
        self.dweights = np.sum(self.X_norm * grads, axis=self.axes,
                               keepdims=self.keepdims)
        # Get derivatives w.r.t. inputs
        self.dinputs = ((1.0 / self.Nt) * self.weights
                        * np.sqrt(self.var + self.epsilon)
                        * (self.Nt * grads - np.sum(grads, axis=self.axes,
                                                    keepdims=self.keepdims)
                        - self.X_mu * (self.var + self.epsilon)**-1
                        * np.sum(grads * self.X_mu, axis=self.axes,
                                 keepdims=self.keepdims)))
        return self.dinputs
