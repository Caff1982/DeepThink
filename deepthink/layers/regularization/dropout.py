import numpy as np

from deepthink.utils import initialize_weights
from deepthink.layers.layer import BaseLayer


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
    def __init__(self, proba, input_shape=None, **kwargs):
        super().__init__(**kwargs)
        self.proba = proba
        self.input_shape = input_shape

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
