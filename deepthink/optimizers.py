import numpy as np


class BaseOptimizer:
    """
    Abstract base class for all optimizers.

    Parameters
    ----------
    lr_decay : float,default=None
        Learning rate decay, the amount to decay the learning rate by
        each epoch. Uses exponential decay:
        - lr = lr0 * lr_decay**iteration
        Suggested values are >=0.9.
    init_lr : float,default=None
        The initial learning rate. This is stored as an attribute to
        enable learning rate decay.
    min_lr : float,default=0.0
        The minimum learning-rate value to use if using learning-rate
        decay.
    iteration : int,default=0
        Counts number of updates, zero indexed.
    layers : list,default=None
        The layers in the network in sequential order.
        This is added during initialization.
    """
    def __init__(self, lr_decay=None, init_lr=None,
                 min_lr=0.0, iteration=0, layers=None):
        self.lr_decay = lr_decay
        self.init_lr = init_lr
        self.min_lr = min_lr
        self.iteration = iteration
        self.layers = layers

    def __repr__(self):
        raise NotImplementedError(
            'All Optimzier subclasses must implement __repr__ method'
        )

    def initialize(self, layers):
        """
        Initialize optimizer to begin training.

        Parameters
        ----------
        layers : list
            A list of layer instances from an instantiated Model
        """
        self.init_lr = self.learning_rate
        self.iteration = 0
        self.layers = layers

    def on_batch_end(self):
        """
        This should be called after each batch-update to increment the
        iteration and update the learning rate.
        """
        self.iteration += 1
        if self.lr_decay:
            self.learning_rate = max(
                self.min_lr,
                np.multiply(self.init_lr,
                            (np.power(self.lr_decay,
                             self.iteration))))

    def update(self):
        raise NotImplementedError(
            'All Optimzier subclasses must implement update method'
        )


class SGD(BaseOptimizer):
    """
    Stochastic Gradient Descent (SGD) optimizer.

    Standard gradient descent update. Also supports optional momentum.

    Parameters
    ----------
    learning_rate : float
        The learning rate determines the step size at each update.
    momentum : float,default=None
        Parameter to dampen oscillations and accelerate updates in the
        relevant direction. Works by applying some of the previous
        updates at each update stage, like a moving average effect.
        Should be in range 0-1 with zero being no momentum added.
    """
    def __init__(self, learning_rate, momentum=None, **kwargs):
        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.momentum = momentum

    def __repr__(self):
        return 'SGD Optimizer'

    def update(self):
        """
        Perform one SGD batch update by looping through layers and
        applying updates to each layer with weights & biases.
        """
        for layer in self.layers:
            # Check that layer has trainable weights
            if hasattr(layer, 'dweights'):
                weight_updates = np.multiply(-self.learning_rate,
                                             layer.dweights)
                if self.momentum:
                    # Add momentum updates
                    weight_updates += np.multiply(self.momentum,
                                                  layer.weight_momentum)
                    # Update layer momentum
                    layer.weight_momentum = weight_updates
                # Perform update to weights
                layer.weights += weight_updates
            # Check if layer has trainable bias
            if hasattr(layer, 'dbiases'):
                bias_updates = np.multiply(-self.learning_rate,
                                           layer.dbiases)
                if self.momentum:
                    bias_updates += np.multiply(self.momentum,
                                                layer.bias_momentum)
                    layer.bias_momentum = bias_updates
                # Perform update to bias
                layer.bias += bias_updates

        # Update learning-rate and iteration
        self.on_batch_end()


class NAG(BaseOptimizer):
    """
    Nesterov Accelerated Gradient (NAG) optimizer.

    This introduces a "look-ahead" by evaluating the gradient at its
    current position plus momentum instead of just at current position.

    Parameters
    ----------
    learning_rate : float
        The learning rate determines the step size of each update.
    momentum : float,default=0.9
        Parameter to dampen oscillations and accelerate updates in the
        relevant direction. Increasing this increases the smoothing at
        each update.

    References
    ----------
    - https://cs231n.github.io/neural-networks-3/#sgd
    """
    def __init__(self, learning_rate, momentum=0.9, **kwargs):
        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.momentum = momentum

    def __repr__(self):
        return 'Nesterov Optimizer'

    def update(self):
        """
        Perform one batch update by looping through layers and applying
        updates to each layer with weights & biases.
        """
        for layer in self.layers:
            # Check that layer has trainable weights
            if hasattr(layer, 'dweights'):
                old_w_mom = layer.weight_momentum.copy()
                # Get the new momentum 
                layer.weight_momentum = np.subtract(
                    np.multiply(self.momentum, old_w_mom),
                    np.multiply(self.learning_rate, layer.dweights))
                # Perform weight update
                layer.weights -= np.subtract(
                    np.multiply(self.momentum, old_w_mom),
                    np.multiply((1 + self.momentum), layer.weight_momentum))
            # Check if layer has trainable bias
            if hasattr(layer, 'dbiases'):
                old_b_mom = layer.bias_momentum.copy()
                layer.bias_momentum = np.subtract(
                    np.multiply(self.momentum, old_b_mom),
                    np.multiply(self.learning_rate, layer.dbiases))
                layer.bias -= np.subtract(
                    np.multiply(self.momentum, old_b_mom),
                    np.multiply((1 + self.momentum), layer.bias_momentum))

        # Update learning-rate and iteration
        self.on_batch_end()


class AdaGrad(BaseOptimizer):
    """
    AdaGrad optimizer.

    AdaGrad uses a gradient cache to store a per parameter sum of
    squared gradients. The gradients are then divided by the square
    root of this average, making this optimizer have an adaptive
    learning rate.

    Parameters
    ----------
    learning_rate : float
        The learning rate determines the step size at each update.
    epsilon : float,default=1e-7
        A constant used to prevent division by zero errors.
    """
    def __init__(self, learning_rate, epsilon=1e-7, **kwargs):
        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.epsilon = epsilon

    def __repr__(self):
        return 'AdaGrad Optimizer'

    def update(self):
        """
        Perform one batch update by looping through layers and applying
        updates to each layer with weights & biases
        """
        for layer in self.layers:
            # Check that layer has trainable weights
            if hasattr(layer, 'dweights'):
                # Normal SGD update
                weight_updates = np.multiply(-self.learning_rate,
                                             layer.dweights)
                bias_updates = np.multiply(-self.learning_rate,
                                           layer.dbiases)
                # Update gradient cache
                layer.weight_grad_cache += np.square(layer.dweights)
                # Perform update and normalize by square-root of cache
                layer.weights += np.divide(
                    weight_updates,
                    np.add(np.sqrt(layer.weight_grad_cache), self.epsilon))
            # Check if layer has trainable bias
            if hasattr(layer, 'dbiases'):
                bias_updates = np.multiply(-self.learning_rate,
                                           layer.dbiases)
                layer.bias_grad_cache += np.square(layer.dbiases)
                layer.bias += np.divide(
                    bias_updates,
                    np.add(np.sqrt(layer.bias_grad_cache), self.epsilon))

        # Update learning-rate and iteration
        self.on_batch_end()


class RMSProp(BaseOptimizer):
    """
    Root Mean Squared Propagation (RMSProp)

    RMSProp uses a gradient cache to store a moving average of squared
    gradients. The updates are then divided by the square root of
    this average. This means the learning rate is adaptive making it
    more robust than standard gradient descent.

    Parameters
    ----------
    learning_rate : float
        The learning rate determines the step size at each update.
    rho : float,default=0.9
        Parameter to control the decay rate at each update.
    epsilon : float,default=1e-7
        A constant used to prevent division by zero errors.
    """
    def __init__(self, learning_rate, rho=0.9, epsilon=1e-7, **kwargs):
        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.rho = rho
        self.epsilon = epsilon

    def __repr__(self):
        return 'RMSProp Optimizer'

    def update(self):
        """
        Perform one batch update by looping through layers and
        applying updates to each layer with weights & biases
        """
        for layer in self.layers:
            # Check that layer has trainable weights
            if hasattr(layer, 'dweights'):
                # Update gradient cache
                layer.weight_grad_cache = np.add(
                    np.multiply(self.rho, layer.weight_grad_cache),
                    np.multiply((1 - self.rho), np.square(layer.dweights)))
                # Perform weight updates
                layer.weights -= np.divide(
                    np.multiply(self.learning_rate, layer.dweights),
                    np.add(np.sqrt(layer.weight_grad_cache), self.epsilon))
            # Check if layer has trainable bias
            if hasattr(layer, 'dbiases'):
                # Update gradient cache
                layer.bias_grad_cache = np.add(
                    np.multiply(self.rho, layer.bias_grad_cache),
                    np.multiply((1 - self.rho), np.square(layer.dbiases)))
                # Perform bias updates
                layer.bias -= np.divide(
                    np.multiply(self.learning_rate, layer.dbiases),
                    np.add(np.sqrt(layer.bias_grad_cache), self.epsilon))

        # Update learning-rate and iteration
        self.on_batch_end()


class Adam(BaseOptimizer):
    """
    Adam optimizer

    Adam combines the idea of an adaptive learning rate with momentum.
    The updates are performed similar to RMSProp however now also use
    momentum updates.

    It also use bias-correction to compensate for initializing the
    momentum and gradient cache arrays with zeros, and therefore the
    tendency to be bias towards zero at the start of training.

    Parameters
    ----------
    learning_rate : float
        The learning rate determines the step size at each update.
    beta1 : float,default=0.9
        Parameter to control the amount of momentum update.
    beta2 : float,default=0.999
        Parameter to control the gradient cache update.
    epsilon : float,default=1e-7
        A constant used to prevent division by zero errors.
    """
    def __init__(self, learning_rate, beta1=0.9, beta2=0.999,
                 epsilon=1e-7, **kwargs):
        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def __repr__(self):
        return 'Adam Optimizer'

    def update(self):
        """
        Perform one batch update by looping through layers and applying
        updates to each layer with weights & biases
        """
        for layer in self.layers:
            # Check that layer has trainable weights
            if hasattr(layer, 'dweights'):
                # Momentum updates
                layer.weight_momentum = np.multiply(
                    self.beta1, layer.weight_momentum) \
                    + np.multiply((1 - self.beta1), layer.dweights)
                # Gradient cache updates
                layer.weight_grad_cache = np.multiply(
                    self.beta2, layer.weight_grad_cache) \
                    + np.multiply((1 - self.beta2), np.square(layer.dweights))
                # Bias correction
                weight_mom_unbias = np.divide(
                    layer.weight_momentum,
                    (1 - np.power(self.beta1, self.iteration+1)))
                weight_grads_unbias = np.divide(
                    layer.weight_grad_cache,
                    (1 - np.power(self.beta2, self.iteration+1)))
                # Perform update on weights
                layer.weights += np.divide(
                    np.multiply(-self.learning_rate, weight_mom_unbias),
                    (np.sqrt(weight_grads_unbias) + self.epsilon))
            # Check if layer has trainable bias
            if hasattr(layer, 'dbiases'):
                # Momentum updates
                layer.bias_momentum = np.multiply(
                    self.beta1, layer.bias_momentum) \
                    + np.multiply((1 - self.beta1), layer.dbiases)
                # Gradient cache updates
                layer.bias_grad_cache = np.multiply(
                    self.beta2, layer.bias_grad_cache) \
                    + np.multiply((1 - self.beta2), np.square(layer.dbiases))
                # Bias correction
                bias_mom_unbias = np.divide(
                    layer.bias_momentum,
                    (1 - np.power(self.beta1, self.iteration+1)))
                bias_grads_unbias = np.divide(
                    layer.bias_grad_cache,
                    (1 - np.power(self.beta2, self.iteration+1)))
                # Perform update on bias
                layer.bias += np.divide(
                    np.multiply(-self.learning_rate, bias_mom_unbias),
                    (np.sqrt(bias_grads_unbias) + self.epsilon))

        # Update learning-rate and iteration
        self.on_batch_end()
