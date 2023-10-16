import numpy as np

from deepthink.layers.layer import BaseLayer


class BasePooling(BaseLayer):
    """Base class for pooling layers."""

    def __init__(self, pool_size=2, stride=2, padding=0, **kwargs):
        super().__init__(**kwargs)
        self.pool_size = pool_size
        self.stride = stride
        self.padding = padding

        # Ensure that either the input shape or the previous layer is set
        if self.prev_layer is None and self.input_shape is None:
            raise ValueError(
                'Pooling layer cannot be used as the first layer'
            )
        # If the input shape is not set, infer it from the previous layer
        if self.input_shape is None:
            self.input_shape = self.prev_layer.output.shape
