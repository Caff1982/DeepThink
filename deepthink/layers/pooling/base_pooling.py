import numpy as np

from deepthink.layers.layer import BaseLayer


class BasePooling(BaseLayer):
    """Base class for pooling layers."""

    def __init__(self, pool_size=2, stride=2, padding=0,
                 input_shape=None, **kwargs):
        super().__init__(**kwargs)
        self.pool_size = pool_size
        self.stride = stride
        self.padding = padding
        self.input_shape = input_shape

    @property
    def input_shape(self):
        # If the _input_shape is not set, infer it from the previous layer
        if self._input_shape is None:
            if self.prev_layer is None:
                raise ValueError('Pooling layer cannot be used as the first layer')
            return self.prev_layer.output.shape
        return self._input_shape

    @input_shape.setter
    def input_shape(self, shape):
        self._input_shape = shape
