import numpy as np


def _set_array_lengths(y_true, y_hat):
    """
    A helper function to resize arrays to the same length.
    If arrays are not the same length then they are resized
    to the minimum length.
    """
    if y_true.shape[0] != y_hat.shape[0]:
        min_len = min(y_true.shape[0], y_hat.shape[0])
        y_true = y_true[:min_len]
        y_hat = y_hat[:min_len]
    return y_true, y_hat


def mean_squared_error(y_true, y_hat):
    """
    Return the Mean Squared Error (MSE) loss between labels
    and predictions. MSE is calculated as the mean of the
    squared differences between the predicted and true values.

    Parameters
    ----------
    y_true : np.array
        The ground truth values
    y_hat : np.array
        The predicted values.

    Returns
    -------
    float
        The MSE loss between predictions and labels.
    """
    y_true, y_hat = _set_array_lengths(y_true, y_hat)
    return np.square(y_true - y_hat).mean()


def root_mean_squared_error(y_true, y_hat):
    """
    Return the Root Mean Squared Error (RMSE) loss between labels
    and predictions. RMSE is the square root of the MSE.

    Parameters
    ----------
    y_true : np.array
        The ground truth values
    y_hat : np.array
        The predicted values.

    Returns
    -------
    float
        The RMSE loss between predictions and labels.
    """
    y_true, y_hat = _set_array_lengths(y_true, y_hat)
    return np.mean(np.sqrt(np.square(y_true - y_hat)))


def mean_absolute_error(y_true, y_hat):
    """
    Return the Mean Absolute Error (MSE) loss between labels
    and predictions. This is also known as the L1 loss.

    Parameters
    ----------
    y_true : np.array
        The ground truth values
    y_hat : np.array
        The predicted values.

    Returns
    -------
    float
        The MAE loss between predictions and labels.
    """
    y_true, y_hat = _set_array_lengths(y_true, y_hat)
    return np.mean(np.abs(y_true - y_hat))


def accuracy(y_true, y_hat):
    """
    Return how often predictions equal labels.

    Both input arrays should be 2D. If the problem is binary
    classification (i.e. y_true.shape[1] == 1), then the predictions
    are rounded to 0 or 1. Otherwise, the predictions are assumed to
    be probabilities from a softmax layer and the index of the highest
    probability is used as the predicted class.

    If arrays are different lengths then they are resized to the
    minimum length.

    Parameters
    ----------
    y_true : np.array
        The ground truth values.
    y_hat : np.array
        The predicted values.

    Returns
    -------
    float
        The accuracy between labels and predictions
    """
    # If arrays are different lengths resize to minimum length
    y_true, y_hat = _set_array_lengths(y_true, y_hat)
    # Check if binary classification
    if y_true.shape[1] == 1:
        return np.mean(np.round(y_true) == np.round(y_hat))
    else:
        # Otherwise, assume multi-class classification
        return np.mean(np.argmax(y_true, axis=1) == np.argmax(y_hat, axis=1))


def confusion_matrix(y_true, y_hat, k):
    """
    Return a 2D confusion matrix where the rows are the actual
    values and columns are predicted values. If arrays are 
    different lengths then they are resized to the minimum length.

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
    y_true, y_hat = _set_array_lengths(y_true, y_hat)
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
