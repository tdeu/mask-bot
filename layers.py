import tensorflow as tf
from tensorflow.keras.layers import Layer

class Cast(Layer):
    def __init__(self, dtype, **kwargs):
        super(Cast, self).__init__(**kwargs)
        self._dtype_value = dtype  # Store in private variable to avoid recursion

    def call(self, inputs):
        return tf.cast(inputs, self._dtype_value)

    def get_config(self):
        config = super(Cast, self).get_config()
        config.update({"dtype": self._dtype_value})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @property
    def dtype(self):
        return self._dtype_value  # Return private variable

    @dtype.setter
    def dtype(self, value):
        self._dtype_value = value 