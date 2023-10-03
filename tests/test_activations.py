import unittest
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from deepthink.activations import Sigmoid, ReLU, LeakyReLU, ELU, TanH, Softmax

# Silencing TensorFlow warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class TestActivations(unittest.TestCase):

    def setUp(self):
        self.inputs = np.array([-10, -1, 0.1, 1, 10])
        self.grads = np.ones_like(self.inputs)

    def test_sigmoid(self):
        self._test_activation(Sigmoid(),
                              tf.keras.activations.sigmoid)

    def test_relu(self):
        self._test_activation(ReLU(),
                              tf.keras.activations.relu)

    def test_leaky_relu(self):
        # Assuming alpha=0.1 for LeakyReLU
        self._test_activation(LeakyReLU(),
                              tf.keras.layers.LeakyReLU(alpha=0.1).call)

    def test_elu(self):
        # Assuming alpha=1.0 for ELU
        self._test_activation(ELU(),
                              tf.keras.activations.elu)

    def test_tanh(self):
        self._test_activation(TanH(),
                              tf.keras.activations.tanh)

    def test_softmax(self):
        # Our Softmax only has forward pass, and also requires inputs
        # to be 2D column vectors.
        input_2D = self.inputs.reshape(1, -1)
        tensor_2D = tf.convert_to_tensor(input_2D)
        expected_outputs = tf.keras.activations.softmax(tensor_2D).numpy()
        np.testing.assert_almost_equal(Softmax().forward(input_2D),
                                       expected_outputs,
                                       decimal=6)

    def _test_activation(self, our_activation, keras_activation):
        self._test_forward(our_activation, keras_activation)
        self._test_backward(our_activation, keras_activation)

    def _test_forward(self, our_activation, keras_activation):
        tensor = tf.convert_to_tensor(self.inputs)
        expected_outputs = keras_activation(tensor).numpy()
        np.testing.assert_almost_equal(our_activation.forward(self.inputs),
                                       expected_outputs,
                                       decimal=6)

    def _test_backward(self, our_activation, keras_activation):
        with tf.GradientTape() as tape:
            inputs_tf = tf.convert_to_tensor(self.inputs, dtype=tf.float32)
            tape.watch(inputs_tf)
            output_tf = keras_activation(inputs_tf)
        expected_dinputs = tape.gradient(output_tf, inputs_tf).numpy()
        np.testing.assert_almost_equal(our_activation.backward(self.grads),
                                       expected_dinputs,
                                       decimal=6)


if __name__ == "__main__":
    unittest.main()
