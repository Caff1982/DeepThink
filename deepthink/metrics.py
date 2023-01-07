import numpy as np


def mean_squared_error(y_true, y_hat):
    """
    Return the Mean Squared Error (MSE) loss between labels and
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
    return np.mean((y_true - y_hat)**2)


def root_mean_squared_error(y_true, y_hat):
    """
    Return the Root Mean Squared Error (RMSE) loss between labels and
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
        The RMSE loss between predictions and labels.
    """
    assert y_true.shape == y_hat.shape
    return np.mean(np.sqrt((y_true - y_hat)**2))


def mean_absolute_error(y_true, y_hat):
    """
    Return the Mean Absolute Error (MSE) loss between labels and
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
        The MAE loss between predictions and labels.
    """
    assert y_true.shape == y_hat.shape
    return np.mean(np.abs(y_true - y_hat))


def accuracy(y_true, y_hat):
    """
    Return how often predictions equal labels.

    Both input arrays should be 2D; y_true should be one-hot-encoded
    and y_hat should be probabilities from softmax layer.

    Parameters
    ----------
    y_true : np.array
        The ground truth values.
    y_hat : np.array
        The predicted values.

    Returns
    -------
    np.array
        The accuracy between labels and predictions
    """
    return np.mean(np.argmax(y_true, axis=1) == np.argmax(y_hat, axis=1))
