import numpy as np


class CategoricalCrossEntropy:
    """
    Categorical Cross Entropy, (CCE).

    Categorical cross-entropy is a loss function that measures the
    difference between the predicted and true class probabilities for
    a set of samples. It is calculated as the negative logarithm of the
    predicted probability of the correct class. It is often used with
    softmax activation functions in classification tasks with multiple
    classes.

    References
    ----------
    - https://en.wikipedia.org/wiki/Cross_entropy
    """

    def __str__(self):
        return 'Categorical Cross-entropy (CCE)'

    def __call__(self, y_true, y_hat):
        return self.loss(y_true, y_hat)

    def loss(self, y_true, y_hat, epsilon=1e-7):
        """
        Return the categorical crossentropy (CCE) loss between labels
        and predictions.

        Labels are expected to be one-hot encoded arrays and
        predictions should be the output of a softmax layer. Both input
        arrays should be the same length. Both predictions and labels
        are stored as instance attributes for use with "grads" method.

        Parameters
        ----------
        y_true : np.array
            The ground truth values, one-hot encoded.
        y_hat : np.array
            The predicted values as probabilties.
        epsilon : float,default=1e-7
            Constant used to avoid log of zero errors.

        Returns
        -------
        cce_loss: float
            The crossentropy loss between predictions and labels
        """

        assert y_true.shape == y_hat.shape
        # Clip values to avoid errors
        y_hat = np.clip(y_hat, epsilon, 1-epsilon)
        # Store labels and predictions for use in backprop
        self.y_true = y_true
        self.y_hat = y_hat
        cce_loss = np.sum(y_true.T * -np.log(y_hat.T)) / y_true.shape[0]
        return cce_loss

    def grads(self):
        """
        Return the gradients/derivative for y_true and y_hat.

        Uses labels and predictions from previous call to "loss"
        to calculate the derivatives.
        """
        return (self.y_hat - self.y_true) / self.y_true.shape[0]


class BinaryCrossEntropy:
    """
    Binary Cross Entropy, (BCE).

    Binary cross-entropy is a loss function that measures the
    difference between the predicted and true probabilities for
    a binary classification task. It is calculated as the negative
    logarithm of the predicted probability of the true class.
    It is often used with sigmoid activation functions in binary
    classification tasks.
    """

    def __str__(self):
        return 'Binary Cross-entropy (BCE)'

    def __call__(self, y_true, y_hat):
        return self.loss(y_true, y_hat)

    def loss(self, y_true, y_hat, epsilon=1e-7):
        """
        Return the binary crossentropy (BCE) loss between labels
        and predictions.

        Labels are expected to be a binary array and predictions
        should be the output of a sigmoid layer. Both input arrays
        should be the same length. Both predictions and labels are
        stored as instance attributes for use with "grads" method.

        Parameters
        ----------
        y_true : np.array
            The ground truth values, binary.
        y_hat : np.array
            The predicted values as probabilties.
        epsilon : float,default=1e-7
            Constant used to avoid log of zero errors.

        Returns
        -------
        bce_loss: float
            The binary crossentropy loss between predictions and labels
        """
        assert y_true.shape == y_hat.shape
        # Clip values to avoid errors
        y_hat_clip = np.clip(y_hat, epsilon, 1-epsilon)
        # Store labels and predictions for use in backprop
        self.y_true = y_true
        self.y_hat = y_hat_clip
        bce_loss = -np.mean(y_true * np.log(y_hat_clip)
                            + (1-y_true) * np.log(1-y_hat_clip))
        return bce_loss

    def grads(self):
        """
        Return the gradients/derivative for y_true and y_hat.

        Uses labels and predictions from previous call to "loss"
        to calculate the derivatives.
        """
        return (-(self.y_true - self.y_hat)
                / (self.y_hat * (1-self.y_hat))
                / self.y_true.shape[0])


class MeanSquaredError:

    def __call__(self, y_true, y_hat):
        return self.loss(y_true, y_hat)[0]

    def __str__(self):
        return 'Mean Squared Error (MSE)'

    def loss(self, y_true, y_hat):
        """
        Return the Mean Squared Error (MSE) loss  between labels and
        predictions. Both arrays should be the same length.

        Parameters
        ----------
        y_true : np.array
            The ground truth values
        y_hat : np.array
            The predicted values.

        Returns
        -------
        np.array
            The MSE loss between predictions and labels.
        """
        assert y_true.shape == y_hat.shape
        self.y_true = y_true
        self.y_hat = y_hat
        self.output = ((y_true - y_hat)**2).mean(axis=0)
        return self.output

    def grads(self):
        return -2 * (self.y_true - self.y_hat) / self.y_true.shape[0]
