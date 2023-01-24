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


def binary_accuracy(y_true, y_hat, threshold=0.5):
    """
    Return how often predictions match binary labels.

    This function provides an accuracy metric for binary
    classification tasks. The predictions, y_hat, should
    be probabilities between 0 and 1. The'threshold' argument
    decides which prediction values are rounded to 0 or 1.

    Parameters
    ----------
    y_true : np.array
        The ground truth values.
    y_hat : np.array
        The predicted values.
    threshold : float,default=0.5
        The threshold for deciding whether prediction values are 1 or 0.

    Returns
    -------
    np.array
        The accuracy between labels and predictions
    """
    assert y_true.shape == y_hat.shape
    return np.mean((y_hat > threshold).astype(int) == y_true)


def confusion_matrix(y_true, y_hat, k):
    """
    Return a 2D confusion matrix where the rows are the actual values
    and columns are predicted values.

    Parameters
    ----------
    y_true : np.array
        The ground truth values
    y_hat : np.array
        The predicted values.
    k : int
        The number of classes.

    Returns
    -------
    result : np.array
        Two dimensional array representing actual and predicted values

    References
    ----------
    - https://en.wikipedia.org/wiki/Confusion_matrix
    """
    # If arrays are one-hot-encoded convert to 1D arrays
    if len(y_true.shape) == 2:
        y_true = np.argmax(y_true, axis=1)
    if len(y_hat.shape) == 2:
        y_hat = np.argmax(y_hat, axis=1)
    # Initialize the empty array with zeros
    result = np.zeros((k, k), dtype=int)
    # Iterate over preds/labels
    for i in range(len(y_hat)):
        result[y_true[i]][y_hat[i]] += 1

    return result
