import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml


def initialize_weights(shape, init_type, dtype=np.float32):
    """
    Return initialized weights for a trainable layer.

    Weights can either be implemented using Glorot or He initialization
    and can be either uniform or normal distribution. Basic uniform
    and normal distribution are also supported.

    Parameters
    ----------
    shape : tuple
        The dimensions of the weights.
    init_type : str
        The initialization type to use.
    dtype : type
        The data-type to use, float32 by default.

    Returns
    -------
    np.array
        A Numpy array of weights

    References
    ----------
    - https://arxiv.org/pdf/1502.01852.pdf
    - http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
    - https://keras.io/api/layers/initializers/
    """
    # Ensure fan_in and fan_out are scalar values
    if len(shape) == 2:
        # Dense layer
        fan_in, fan_out = shape
    else:
        # Convolutional kernels, shape:
        # (depth-out, depth-in, kernel, kernel)
        receptive_field = np.prod(shape[2:])
        fan_in = shape[1] * receptive_field
        fan_out = shape[0] * receptive_field

    if init_type == 'glorot_normal':
        stddev = np.sqrt(2.0 / (fan_in + fan_out))
        return np.random.normal(0.0, stddev, size=shape).astype(dtype)
    elif init_type == 'glorot_uniform':
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        return np.random.uniform(limit, -limit, size=shape).astype(dtype)
    elif init_type == 'he_normal':
        stddev = np.sqrt(2.0 / fan_in)
        return np.random.normal(0.0, stddev, size=shape).astype(dtype)
    elif init_type == 'he_uniform':
        limit = np.sqrt(6.0 / fan_in)
        return np.random.uniform(limit, -limit, size=shape).astype(dtype)
    elif init_type == 'uniform':
        return np.random.uniform(-0.05, 0.05, size=shape).astype(dtype)
    elif init_type == 'normal':
        return np.random.normal(0.0, 0.05, size=shape).astype(dtype)
    else:
        raise Exception(f'Weight init type "{init_type}" not recognized')


def get_strided_view_1D(arr, view_shape, stride):
    """
    Return a view of an array using Numpy's as_strided
    slide-trick.

    Computationally efficient way to get all the kernel windows
    to be used in the convolution operation. Takes 3D tensor as
    input with shape (batch, depth, seq-len) and outputs 4D tensor
    with shape (batch, seq-len, depth, kernel-size).

    Parameters
    ----------
    arr : np.array
        The array/tensor to perform the operation on.
    view_shape : tuple
        The shape of the view to be returned.
    stride : int
        The step size between each view.

    Returns
    -------
    view : np.array
        The 4D view to be used in forward/backward pass.
    """
    # strides returns the byte-step for each dim in memory
    s0, s1, s2 = arr.strides
    strides = (s0, stride * s2, s1, s2)
    # Return a view of the array with the given shape and strides
    return np.lib.stride_tricks.as_strided(
        arr, view_shape, strides=strides, writeable=True)


def get_strided_view_2D(arr, view_shape, stride):
    """
    Return a view of an array using Numpy's as_strided
    slide-trick.

    Computationally efficient way to get all the kernel windows
    to be used in the convolution operation. Takes 4D tensor with
    shape (batch, depth, img-size, img-size) as input and outputs
    a 6D tensor with shape:
    (batch, img-size, img-size, depth, kernel-size, kernel-size).

    Parameters
    ----------
    arr : np.array
        The array/tensor to perform the operation on.
    view_shape : tuple
        The shape of the view to be returned.
    stride : int
        The step size between each view.

    Returns
    -------
    view : np.array
        The 6D view to be used in forward/backward pass
    """
    # strides returns the byte-step for each dim in memory
    s0, s1, s2, s3 = arr.strides
    strides = (s0, stride * s2, stride * s3, s1, s2, s3)
    # Return a view of the array with the given shape and strides
    return np.lib.stride_tricks.as_strided(
        arr, view_shape, strides=strides, writeable=True)


def get_strided_view_3D(arr, view_shape, stride):
    """
    Return a view of an array using Numpy's as_strided
    slide-trick.

    Computationally efficient way to get all the kernel windows
    to be used in the convolution operation. Takes 5D tensor with
    shape (batch, depth, img-size, img-size, img-size) as input and
    outputs a 8D tensor with shape:
    (batch, img-size, img-size, img-size, depth, kernel-size,
    kernel-size, kernel-size).

    Parameters
    ----------
    arr : np.array
        The array/tensor to perform the operation on.
    view_shape : tuple
        The shape of the view to be returned.
    stride : int
        The step size between each view.

    Returns
    -------
    view : np.array
        The 8D view to be used in forward/backward pass
    """
    # strides returns the byte-step for each dim in memory
    s0, s1, s2, s3, s4 = arr.strides
    strides = (s0, stride * s2, stride * s3, stride * s4, s1, s2, s3, s4)
    # Return a view of the array with the given shape and strides
    return np.lib.stride_tricks.as_strided(
        arr, view_shape, strides=strides, writeable=True)


def pad_1D(arr, padding, mode='constant'):
    """
    Return array with padding added to sequence dimension.

    Parameters
    ----------
    arr : np.array
        The array to perform the operation on.
    padding : tuple
        The amount of padding to add to the sequence dimension.

    Returns
    -------
    padded : np.array
        The array with padding added to sequence dimension.
    """
    return np.pad(
        arr,
        pad_width=((0, 0), (0, 0),
                   padding),
        mode=mode
    )


def pad_2D(arr, padding, mode='constant'):
    """
    Return array with padding added to height & width dimensions.

    Parameters
    ----------
    arr : np.array
        The array to perform the operation on.
    padding : tuple
        The amount of padding to add to the height & width dimensions.

    Returns
    -------
    padded : np.array
        The array with padding added to height & width dimensions.
    """
    return np.pad(
        arr,
        pad_width=((0, 0), (0, 0),
                   padding,
                   padding),
        mode=mode
    )


def one_hot_encode(arr,  k, dtype=np.float32):
    """
    Takes 1D array of target integer values and return a
    one-hot-encoded 2D array.

    Parameters
    ----------
    arr : np.array
        The 1D array to encode
    k : int
        The number of target classes
    dtype : type,default=np.float32
        The data-type to use

    Returns
    -------
    np.array
        The one-hot-encoded array
    """
    return np.eye(k, dtype=dtype)[arr]


def pad_sequences(sequences, maxlen):
    """
    Pad sequences to a fixed length maxlen.

    Any sequences longer than maxlen will be truncated.
    Any sequences shorter than maxlen will be padded with zeros.

    Parameters
    ----------
    sequences : list
        List of sequences to pad.
    maxlen : int
        Maximum length of each sequence.

    Returns
    -------
    padded_sequences : np.ndarray
        Padded sequences.
    """
    padded_sequences = np.zeros((len(sequences), maxlen), dtype=np.int32)
    for i, sequence in enumerate(sequences):
        # Truncate sequences that are too long
        if len(sequence) > maxlen:
            sequence = sequence[:maxlen]
        padded_sequences[i, :len(sequence)] = sequence
    return padded_sequences


def load_mnist_data(filepath=None, test_split=60000,
                    shuffle=True, flat_data=False):
    """
    A function to load and prepare the MNIST dataset.

    The data is split into 60k training & 10k testing samples by
    default. Data values are normalized to range 0 to 1. Labels are
    one-hot-encoded.

    Filepath argument can be used to save/load the dataset on the
    local machine.

    Parameters
    ----------
    filepath : str,default=None
        The location to load/save the data. When this argument is used
        it will attempt to load the dataset from the filepath. If it
        does not already exist it will download the dataset and save to
        this location as a pickle file.
    test_split : int
        The number of training samples, everything after this
        returned as test samples.
    shuffle : bool,default=True
        Used to shuffle training data, True by default.
    flat_data : bool,default=False
        When set to True images are represented by 1D arrays of length
        784, when set to False images are reshaped to image dimensions:
        (1, 28, 28).

    Returns
    -------
    training_data : tuple
        A tuple of (X_train, y_train), both numpy arrays, for training
        X-data and labels.
    test_data : tuple
        A tuple of (X_test, y_test), both numpy arrays, for test X-data
        and labels.
    """
    if filepath:
        # Download dataset if file does not already exist
        if not os.path.exists(filepath):
            print('Downloading dataset...')
            dataset = fetch_openml('mnist_784')
            pickle.dump(dataset, open(filepath, 'wb'))
        else:
            dataset = pickle.load(open(filepath, 'rb'))
    else:
        # If no filepath argument download the dataset
        print('Downloading dataset...')
        dataset = fetch_openml('mnist_784')

    # Get X & y data and convert data-type
    X = dataset['data'].values.astype(np.float32)
    y = dataset['target'].values.astype(np.int8)

    # Normalize image values to 0-1
    X /= 255.0
    # One-hot encode labels and convert to integer
    y_new = one_hot_encode(y, k=10, dtype=np.int8)
    # Split data into training and testing
    X_train, y_train = X[:test_split], y_new[:test_split]
    X_test, y_test = X[test_split:], y_new[test_split:]

    if shuffle:
        # Shuffle training data
        shuffle_index = np.random.permutation(test_split)
        X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]
    if not flat_data:
        # Reshape from 1D to image dimensions, channels first
        X_train = X_train.reshape(-1, 1, 28, 28)
        X_test = X_test.reshape(-1, 1, 28, 28)

    training_data = (X_train, y_train)
    test_data = (X_test, y_test)
    return training_data, test_data


def display_confusion_matrix(matrix, k, title=None, figsize=(10, 10)):
    """
    Display a confusion matrix showing predctions and labels.

    Parameters
    ----------
    matrix : np.array
        A 2D confusion matrix array where rows are predicted labels and
        columns are actual labels.
    k : int
        The number of classes
    title : str,default=None
        Optional argument to display a title on the visualization.
    figsize : tuple,default=(10, 10)
        The Matplotlib figure-size to use.
    """
    fig, ax = plt.subplots(figsize=figsize)
    # Use imshow to display the values as colors
    ax.imshow(matrix, cmap='turbo')
    # Iterate over each column/row and add value from matrix as text
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            ax.text(j, i, matrix[i, j], color='w', size=15,
                    ha='center', va='center')
    # Setting the x & y ticks
    ticks = list(range(k))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_ylabel('Predicted label', fontsize=15)
    ax.set_xlabel('Actual label', fontsize=15)

    plt.grid(False)
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()
