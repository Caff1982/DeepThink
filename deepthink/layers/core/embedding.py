import numpy as np

from deepthink.utils import initialize_weights
from deepthink.layers.layer import BaseLayer


class Embedding(BaseLayer):
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
        self.input_shape = input_shape
        self.seq_len = input_shape[-1]
        self.weight_init = weight_init
        

    def __str__(self):
        return 'Embedding Layer'

    def initialize(self):
        """
        Initialize the embedding layer.
        """
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
