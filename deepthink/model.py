import numpy as np
from tqdm import tqdm

from deepthink.history import History
from deepthink.layers import Dropout, BatchNorm


class Model:
    """
    Neural network model class.

    This is used to construct and train the neural network. The layers
    are contained in a list, self.layers, created by calling add_layer.
    Once the layers have been created the network can be initialized by
    calling initialize. The API is designed to be similar to Keras.

    Parameters
    ----------
    optimizer : (any optimizer from optimizers.py)
        The optimizer to use to perform updates.
    cost : (cost function)
        The cost/loss function to use.
    batch_size : int,default=64
        The number of samples in each mini-batch at each update
    metrics : list,default=None
        The metrics to use when recording training history.
        The cost/loss function will automatically be added to this,
        other metrics can be added and must match the keys contained
        in History object's metric_dict.
    dtype : type,default=np.float32
        The numpy data-type to be used. Default is np.float32,
        np.float64 is required for gradient checking
    """

    def __init__(self, optimizer, cost, batch_size=64,
                 metrics=None, dtype=np.float32):
        self.optimizer = optimizer
        self.cost = cost
        self.batch_size = batch_size
        if metrics is None:
            # If no metics supplied use the cost function only
            self.metrics = [self.cost]
        else:
            self.metrics = metrics
            # Insert cost function to be first in metrics
            self.metrics.insert(0, self.cost)

        self.dtype = dtype
        # Initialize layers array to store the network's layers
        self.layers = []

    def initialize(self):
        """
        Initialize model parameters before training.

        For each layer in the network previous and next layer
        attributes are created to enable backpropagation between
        layers. Each layer is also initialized to prepare for training.
        Finally, the optimizer is initialized and the layers array can
        then be used by it to make updates.
        """
        # Set next/previous layer attributes
        for i, layer in enumerate(self.layers):
            if i == 0:
                # First layer, no previous layer
                layer.next_layer = self.layers[i+1]
            elif i == len(self.layers)-1:
                # Last layer, no next_layer
                layer.prev_layer = self.layers[i-1]
            else:
                # Hidden layer
                layer.prev_layer = self.layers[i-1]
                layer.next_layer = self.layers[i+1]
            # Set data-type for layer
            layer.dtype = self.dtype

            # Initialize parameters for layer
            if hasattr(layer, 'initialize'):
                # All layers have initialize except activations
                layer.initialize()
            else:
                # Activation function so outputs same shape as inputs
                layer.input_shape = self.layers[i-1].output.shape
                layer.output = np.zeros(layer.input_shape, dtype=self.dtype)
        # Set layers array attribute in optimizer
        self.optimizer.initialize(self.layers)

    def summary(self, line_length=65):
        """
        Print a summary of the model.

        Parameters
        ----------
        line_length : int,default=65
            The maximum length of printed lines
        """
        n_params = len(self.get_params())
        headers = ['Layer Type', 'Output Shape', 'Param #']

        print('Model summary:')
        print('_' * line_length)
        print(f'{headers[0]} {headers[1]:>28} {headers[2]:>25}')
        print('=' * line_length)
        # Iterate through layers and print a row for each
        for layer in self.layers:
            output_shape = (None, *layer.output.shape[1:])
            if hasattr(layer, 'weights'):
                row = [layer, output_shape, layer.weights.size]
                if hasattr(layer, 'bias'):
                    row[2] += layer.bias.size
            else:
                row = [layer, output_shape, 0]
            row = [str(x) for x in row]
            print(f'{row[0]:<25} {row[1]:<20} {row[2]:>18s}')
        print('=' * line_length)
        print(f'Total params: {n_params:,}')

    def add_layer(self, layer):
        """
        Add a layer instance to the layers array.

        Parameters
        ----------
        layer : (any layer subclass of BaseLayer)
            The instantiated layer to add to the model.
        """
        self.layers.append(layer)

    def forward(self, X, training=True):
        """
        Perform a forward pass through the network layers.

        Parameters
        ---------
        X : numpy.array
            The batch of input data to perform forward pass.
        training : bool,default=True
            Parameter used for Dropout/BatchNorm layers since they
            operate differently between training and inference.

        Returns
        -------
        numpy.array
            The output predictions from the model.
        """
        for layer in self.layers:
            if isinstance(layer, (BatchNorm, Dropout)):
                X = layer.forward(X, training=training)
            else:
                X = layer.forward(X)
        return X

    def backward(self, dZ):
        """
        Perform backpropagation.

        The error is backpropagated from the output layer to the input
        layer. Backward method is called on each layer which calculates
        the gradient/derivative estimates w.r.t inputs. This is
        propogated through the network using the chain-rule.

        Parameters
        ----------
        dZ : numpy.array
            The gradients from the models output predictions,
            calculated by calling cost.grads().
        """
        # Check last activation layer has backward method
        if hasattr(self.layers[-1], 'backward'):
            self.layers[-1].backward(dZ)
        else:
            # If no backward method must be Softmax
            self.layers[-1].dinputs = dZ

        for layer in reversed(self.layers[:-1]):
            layer.backward(layer.next_layer.dinputs)

    def train(self, training_data, validation_data,
              epochs=10, verbose=True, shuffle=True):
        """
        Used to train the model for a number of epochs.

        Model performance is recorded after each epoch and returned in
        a History object which contains a dict, 'history', mapping
        metrics to their performance each epoch. This is similar to
        Keras callbacks.History.history.

        Parameters
        ----------
        training_data : tuple
            The data to perform the training on. This should be a tuple
            of numpy arrays with shape (X-data, y-data).
        validation_data : tuple
            The data used for validating the model. This should be a
            tuple of numpy arrays with shape (X-data, y-data).
        epochs : int,default=10
            The number of epochs to train on the dataset.
        verbose : bool,default=True
            Controls verbosity mode. If True a summary will be printed
            each epoch, if set to False nothing is printed.
        shuffle : bool,default=True
            When set to True training data is shuffled every epoch.

        Returns
        -------
        history : History
            A history object containing the model's performance for
            each metric on training and validation data.
        """
        # Unpack X-data and target y-values
        X_train, y_train = training_data
        X_val, y_val = validation_data
        # Get the number of training batches per epoch
        num_batches = (X_train.shape[0] // self.batch_size)
        # initialize history dict to track training progress per epoch
        self.history = History(self.metrics, verbose=verbose, n_epochs=epochs)

        for epoch in range(epochs):
            if shuffle:
                # Shuffle training data
                permutation = np.random.permutation(X_train.shape[0])
                X_train_shuffled = X_train[permutation]
                y_train_shuffled = y_train[permutation]
            # initialize model_outputs to store model predictions and
            # is used to evaluate performance at epoch end.
            model_outputs = np.zeros((num_batches * self.batch_size,
                                     y_train.shape[1]), dtype=self.dtype)
            # Iterate over each mini-batch
            for batch_idx in tqdm(range(num_batches), disable=not verbose):
                start = batch_idx * self.batch_size
                end = start + self.batch_size
                X_batch = X_train_shuffled[start:end]
                y_batch = y_train_shuffled[start:end]
                # Forward pass
                outputs = self.forward(X_batch)
                # Store outputs to be used for evaluation
                model_outputs[start:end] = outputs
                # Get loss & derivatives and perform backward pass
                loss = self.cost.loss(y_batch, outputs)
                dZ = self.cost.grads()
                self.backward(dZ)
                # Optimizer update
                self.optimizer.update()

            # Evaluate performance
            if epoch == 0:
                # On first epoch get new predictions to avoid
                # reporting large error during first few batches
                train_preds = self.get_predictions(X_train_shuffled)
            else:
                train_preds = model_outputs

            train_labels = y_train_shuffled[:model_outputs.shape[0]]
            val_preds = self.get_predictions(X_val)
            val_labels = y_val[:val_preds.shape[0]]
            # Update history
            self.history.on_epoch_end(train_labels, train_preds,
                                      val_labels, val_preds)
        return self.history

    def get_predictions(self, X):
        """
        Get the models predictions for a batch of data.

        Parameters
        ----------
        X : numpy.array
            The input data to make predictions on.

        Returns
        -------
        predictions : numpy.array
            The model's predictions as a 1D array.
        """
        len_X = X.shape[0]
        len_preds = len_X - (len_X % self.batch_size)

        if len_X < self.batch_size:
            # If number of samples is less than batch-size
            # create a zero padded array of length batch-size
            if self.layers[0].input_shape:
                # Conv2D layer
                padded_X = np.zeros(self.layers[0].input_shape)
            else:
                # Dense layer
                padded_X = np.zeros((self.batch_size,
                                     self.layers[0].n_inputs))
            padded_X[:len_X] = X
            X = padded_X
            # set prediction length to be batch-size
            len_preds = self.batch_size

        # Create the array to store predictions
        predictions = np.zeros((len_preds, self.layers[-1].output.shape[1]))
        # batch_idx is used as current batch index
        batch_idx = 0
        while batch_idx < len_preds:
            batch_preds = self.forward(X[batch_idx:batch_idx + self.batch_size],
                                       training=False)
            predictions[batch_idx:batch_idx + self.batch_size] = batch_preds
            batch_idx += self.batch_size

        if len_X < self.batch_size:
            return predictions[:len_X]
        else:
            return predictions

    def get_params(self):
        """
        Return model weights & biases for each layer
        unrolled into a 1D vector.
        """
        params = []
        for layer in self.layers:
            if hasattr(layer, 'weights'):
                params.extend(layer.weights.flatten())
            if hasattr(layer, 'bias'):
                params.extend(layer.bias.flatten())
        return np.array(params)

    def set_params(self, params):
        """
        Takes 1D vector and uses that to set model
        weights and biases
        """
        idx = 0
        for layer in self.layers:
            if hasattr(layer, 'weights'):
                weights_arr = params[idx:idx+layer.weights.size]
                layer.weights = weights_arr.reshape(layer.weights.shape)
                idx += layer.weights.size
            if hasattr(layer, 'bias'):
                bias_arr = params[idx:idx+layer.bias.size]
                layer.bias = bias_arr.reshape(layer.bias.shape)
                idx += layer.bias.size

    def save(self, filepath):
        """
        Save the model's weights & biases to specified filepath.

        File-type should be '.npy'.

        Parameters
        ----------
        filepath : str
            The location to save the model's parameters.
        """
        params = self.get_params()
        np.save(filepath, params)

    def load(self, filepath):
        """
        Load model parameters from specified filepath.

        Parameters
        ----------
        filepath : str
            The location to load the model's parameters from.
        """
        params = np.load(filepath)
        self.set_params(params)
