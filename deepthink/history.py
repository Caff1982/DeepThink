import time

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from deepthink.metrics import (mean_squared_error,
                               root_mean_squared_error,
                               mean_absolute_error,
                               accuracy)


class History:
    """
    A class to record model training history.

    Initialized when Model.train is called, this class records the
    model's performance after each epoch on train and validation
    predictions. It will store the model's loss/cost automatically,
    other metrics can be added using Model's 'metrics' argument.

    The metrics are stored in a dictionary, 'history', where the
    keys are the metrics and values are arrays recording performance
    at each epoch. This is similar to Kera's History.history.

    Training history can be visualized by calling 'plot_history'
    once training has been completed.

    Parameters
    ----------
    metrics : list
        A list of the metrics to use, created during model
        initialization. This should always contain the model's
        cost/loss as first element. Other metrics are optional
        and must be keys of the 'metric_dict' class attribute.
    verbose : bool,default=True
        Controls verbosity mode. If True a summary will be printed
        each epoch, if set to False nothing is printed.
    n_epochs : int,default=None
        The number of training epochs, used for displaying updates.
        This is a required argument when verbose is set to True.

    Attributes
    ----------
    metric_dict : dict
        A dictionary which maps strings to metric loss function.
    """
    metric_dict = {
        'accuracy': accuracy,
        'RMSE': root_mean_squared_error,
        'MAE': mean_absolute_error,
        'MSE': mean_squared_error,
        }

    def __init__(self, metrics, n_epochs, verbose=True):
        self.metrics = metrics
        self.n_epochs = n_epochs
        self.verbose = verbose
        
        if self.verbose:
            # Store training time for displaying updates
            self.start_time = time.time()

        self.history = {}
        # Add 'loss' as default first metric
        self.history['loss'] = []
        self.history['val_loss'] = []
        for metric in self.metrics[1:]:
            # Add other optional metrics
            self.history[metric] = []
            self.history[f'val_{metric}'] = []

    def on_epoch_end(self, y_train, train_preds, y_val, val_preds):
        """
        This method calculates the current performance on all metrics
        and stores them in the history dictionary. When verbose is set
        to True an update is printed on screen.

        Parameters
        ----------
        y_train : np.array
            The y-target training values
        train_preds : np.array
            The model's training predictions
        y_val : np.array
            The y-target validation values
        val_preds : np.array
            The model's validation predictions
        """
        # Get and store the model's loss/cost function
        train_loss = self.metrics[0](y_train, train_preds)
        val_loss = self.metrics[0](y_val, val_preds)
        self.history['loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)

        for metric in self.metrics[1:]:
            # Get train & validation value for each metric
            train_value = self.metric_dict[metric](y_train, train_preds)
            val_value = self.metric_dict[metric](y_val, val_preds)
            self.history[metric].append(train_value)
            self.history[f'val_{metric}'].append(val_value)

        if self.verbose:
            # Add epoch progress and elapsed time to row variable
            elapsed_time = time.time() - self.start_time
            current_epoch = len(self.history['loss'])
            row = f"Epoch: {current_epoch}/{self.n_epochs}, " + \
                  f"elapsed-time: {elapsed_time:.2f}s - "

            # Add training cost/loss and all other metrics to row
            row += f"loss: {self.history['loss'][-1]:.4f} - "
            for metric in self.metrics[1:]:
                last_value = self.history[metric][-1]
                row += f"{metric}: {last_value:.4f} - "
            # Add validation loss and metrics to row
            row += f"val_loss: {self.history['val_loss'][-1]:.4f} - "
            for metric in self.metrics[1:]:
                last_value = self.history['val_' + metric][-1]
                row += f"val_{metric}: {last_value:.4f} - "
            # Display row
            print(row)

    def plot_history(self, display_image=True, save_fname=None):
        """
        Display model training performance at each epoch.

        Plots model cost/loss per epoch on training and validation.
        If any additional metrics are added then it will plot loss
        and one other metric.

        Parameters
        ----------
        display_image : book,default=True
            Boolean parameter to control whether the images is shown
            or not
        save_fname : str,default=None
            Optional argument to save the image, if used it should be
            the filename to save the image as.
        """
        # Set figure-size depending on number of metrics
        if len(self.metrics) == 1:
            fig, axes = plt.subplots(1, figsize=(14, 8))
            # Need to be able to index into axes
            axes = [axes]
        else:
            fig, axes = plt.subplots(2, figsize=(14, 12))

        x_labels = list(range(len(self.history['loss'])))

        # Plot the model's loss performance
        axes[0].plot(self.history['loss'], label='Train loss')
        axes[0].plot(self.history['val_loss'], label='Val loss')
        axes[0].set_ylabel('Loss', fontsize='x-large')
        axes[0].legend(fontsize='large', framealpha=1, fancybox=True)
        axes[0].set_xticks(ticks=x_labels, labels=x_labels)
        # MaxNLocator used to dynamically set xtick locations
        axes[0].xaxis.set_major_locator(MaxNLocator(20))

        if len(self.metrics) > 1:
            # Plot additional metric if included
            metric = self.metrics[1]
            axes[1].plot(self.history[metric], label=f'Train {metric}')
            axes[1].plot(self.history[f'val_{metric}'], label=f'Val {metric}')
            axes[1].set_ylabel(metric.capitalize(), fontsize='x-large')
            axes[1].legend(fontsize='large', framealpha=1, fancybox=True)
            axes[1].set_xticks(ticks=x_labels, labels=x_labels)
            axes[1].xaxis.set_major_locator(MaxNLocator(20))

        plt.xlabel('Epoch', fontsize='large')
        plt.legend(fontsize='large', framealpha=1, fancybox=True)
        if save_fname:
            try:
                plt.savefig(save_fname)
            except Exception as e:
                print(f'Failed to save image: {e}')
        if display_image:
            plt.show()
