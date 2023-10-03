import unittest
import numpy as np

from tensorflow.keras.losses import MeanSquaredError as keras_MSE
from tensorflow.keras.losses import BinaryCrossentropy as keras_BCE
from tensorflow.keras.losses import CategoricalCrossentropy as keras_CCE
import tensorflow as tf

from deepthink.loss import (
    MeanSquaredError,
    BinaryCrossEntropy,
    CategoricalCrossEntropy
)
from deepthink.activations import Softmax

# Silencing TensorFlow warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class TestLosses(unittest.TestCase):

    def setUp(self):
        # For MeanSquaredError
        self.y_true_mse = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.y_hat_mse = np.array([1.5, 2.5, 2.5, 3.5, 4.5])

        # For BinaryCrossEntropy
        self.y_true_bce = np.array([0, 1, 0, 1, 0])
        self.y_hat_bce = np.array([0.1, 0.2, 0.2, 0.3, 0.2])

        # For CategoricalCrossEntropy (assuming binary classes for simplicity)
        self.y_true_ce = np.array([[1, 0], [0, 1], [1, 0], [0, 1], [1, 0]])
        self.y_hat_ce_logits = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [5, 10]])
        self.y_hat_ce_softmax = np.array([[0.26894142, 0.73105858],
                                        [0.26894142, 0.73105858],
                                        [0.26894142, 0.73105858],
                                        [0.26894142, 0.73105858],
                                        [0.00669285, 0.99330715]])

    def test_mean_squared_error(self):
        self._test_loss(MeanSquaredError(), keras_MSE(),
                        self.y_true_mse, self.y_hat_mse)

    def test_binary_cross_entropy(self):
        self._test_loss(BinaryCrossEntropy(), keras_BCE(),
                        self.y_true_bce, self.y_hat_bce)

    def test_categorical_cross_entropy(self):
        self._test_loss(CategoricalCrossEntropy(), keras_CCE(from_logits=True),
                        self.y_true_ce, self.y_hat_ce_logits)

    def _test_loss(self, our_loss, keras_loss, y_true, y_hat):
        self._test_loss_value(our_loss, keras_loss, y_true, y_hat)
        self._test_loss_grads(our_loss, keras_loss, y_true, y_hat)

    def _test_loss_value(self, our_loss, keras_loss, y_true, y_hat):
        y_true_tf = tf.convert_to_tensor(y_true, dtype=tf.float32)
        y_hat_tf = tf.convert_to_tensor(y_hat, dtype=tf.float32)
        expected_loss = keras_loss(y_true_tf, y_hat_tf).numpy()

        if isinstance(our_loss, CategoricalCrossEntropy):
            y_hat_sm = Softmax().forward(y_hat)
            np.testing.assert_almost_equal(our_loss.loss(y_true, y_hat_sm),
                                           expected_loss, decimal=6)
        else:
            np.testing.assert_almost_equal(our_loss.loss(y_true, y_hat),
                                           expected_loss, decimal=6)

    def _test_loss_grads(self, our_loss, keras_loss, y_true, y_hat):
        y_true_tf = tf.convert_to_tensor(y_true, dtype=tf.float32)
        y_hat_tf = tf.Variable(y_hat, dtype=tf.float32)
        with tf.GradientTape() as tape:
            loss_tf = keras_loss(y_true_tf, y_hat_tf)
        expected_grads = tape.gradient(loss_tf, y_hat_tf).numpy()
        np.testing.assert_almost_equal(our_loss.grads(),
                                       expected_grads,
                                       decimal=6)


if __name__ == "__main__":
    unittest.main()
