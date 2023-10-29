import numpy as np


class Sigmoid:
    """
    Sigmoid activation function: "sigmoid(x) = 1 / 1 (1 + exp(-x))".

    Applies the sigmoid activation function to input. Output values
    are bounded between 0 and 1. Negative input values return <0.5
    and positive inputs values return >0.5.

    Sigmoid is often used as the output activation for binary
    classification problems and can be used in regression tasks.
    It is rarely used as an activation function between layers
    due to the saturating/vanishing gradients problem.

    References
    ----------
    - https://en.wikipedia.org/wiki/Sigmoid_function
    - https://beckernick.github.io/sigmoid-derivative-neural-network/
    """
    def __repr__(self):
        return 'Sigmoid'

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Return the input array with sigmoid activation applied.

        The output is also stored as an instance attribute to be
        used in backpropagation.
        """
        self.output = 1.0 / (1.0 + np.exp(-x))
        return self.output

    def backward(self, grads):
        """
        Return the derivative of sigmoid function.

        The result is stored as an instance attribute "dinputs",
        to be used in backpropagation.
        """
        self.dinputs = grads * (1 - self.output) * self.output
        return self.dinputs


class ReLU:
    """
    Rectified linear unit (ReLU) activation function.

    Applies elementwise "max(x, 0)", i.e. negative values returned
    as zero while positive values remain unchanged.

    References
    ----------
    - https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
    - https://arxiv.org/abs/1803.08375 (ReLU paper)
    """
    def __repr__(self):
        return 'ReLU'

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Return the input array with ReLU activation applied.

        The inputs are also stored as instance attributes to be
        used in backpropagation.
        """
        self.inputs = x
        self.output = np.maximum(0, x)
        return self.output

    def backward(self, grads):
        """
        Return the derivative of ReLU function.

        The result is stored as an instance attribute "dinputs",
        to be used in backpropogation.
        """
        self.dinputs = grads.copy()
        self.dinputs[self.inputs < 0] = 0.0
        return self.dinputs


class LeakyReLU:
    """
    Leaky version of Rectified Linear Unit activation function.

    Applies a small gradient when the values are negative, otherwise
    returns the value, applied elementwise.

    Parameters
    ----------
    alpha : float,default=0.1
        Constant to scale negative values by. Controls the slope
        of the negative gradients.

    References
    ----------
    - https://en.wikipedia.org/wiki/Rectifier_(neural_networks)#Variants
    - https://arxiv.org/pdf/1505.00853.pdf
    """
    def __init__(self, alpha=0.1):
        self.alpha = alpha

    def __repr__(self):
        return 'Leaky-ReLU'

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Return the input array with Leaky ReLU activation applied.

        The inputs are also stored as instance attributes to be
        used in backpropagation.
        """
        self.inputs = x
        self.output = np.where(x < 0, x * self.alpha, x)
        return self.output

    def backward(self, grads):
        """
        Return the derivative of Leaky ReLU function.

        The result is stored as an instance attribute "dinputs",
        to be used in backpropogation.
        """
        self.dinputs = np.where(self.inputs < 0, grads * self.alpha, grads)
        return self.dinputs


class ELU:
    """
    Exponential Linear Unit activation function.

    Parameters
    ----------
    alpha : float,default=1.0
        Scale for the negative factor. Decreasing this will make the
        activation more like ReLU.

    References
    ----------
    - https://en.wikipedia.org/wiki/Rectifier_(neural_networks)#Variants
    """
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def __repr__(self):
        return 'ELU'

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Return the input array with ELU activation applied.

        The inputs are also stored as instance attributes to be
        used in backpropagation.
        """
        self.inputs = x
        self.output = np.where(x < 0, self.alpha * (np.exp(x) - 1.0), x)
        return self.output

    def backward(self, grads):
        """
        Return the derivative of ELU function.

        The result is stored as an instance attribute "dinputs",
        to be used in backpropogation.
        """
        self.dinputs = np.where(self.inputs < 0,
                                grads * self.alpha * np.exp(self.inputs),
                                grads)
        return self.dinputs


class TanH:
    """
    Hypberbolic Tangent (TanH) activation function.

    Output values are bounded between -1 and 1 and centered at zero.

    References
    ----------
    - https://mathworld.wolfram.com/HyperbolicTangent.html
    """
    def __repr__(self):
        return 'TanH'

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Return the input array with tanh activation applied.

        The output is also stored as an instance attributes
        to be used in backpropagation.
        """
        self.output = np.tanh(x)
        return self.output

    def backward(self, grads):
        """
        Return the derivative of tanh function.

        The result is stored as an instance attribute "dinputs",
        to be used in backpropogation.
        """
        self.dinputs = grads * (1.0 - np.square(self.output))
        return self.dinputs


class Softmax:
    """
    Softmax converts a vector of class predictions/logits to a
    probability distribution where all values are in range (0, 1)
    and the values sum to one.

    Softmax is often used as the last layer in multi-class
    classification problems where a probability distribution
    representing each class is required.

    N.B. This activation function has no backward method. This is
    is because it is not needed: the gradients/dZ that categorical
    cross-entropy cost function returns are the derivatives for this
    activation function.

    References
    ----------
    -https://en.wikipedia.org/wiki/Softmax_function
    """
    def __repr__(self):
        return 'Softmax'

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Return the array with softmax activation applied.

        The modified array is stored as an instance attribute
        output to be used in backpropagation.
        """
        # Max value subtracted to avoid NaNs
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.output = exps / np.sum(exps, axis=1, keepdims=True)
        return self.output
