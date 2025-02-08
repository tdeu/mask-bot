import tensorflow as tf
from tensorflow.keras.layers import Layer

class Cast(Layer):
    def __init__(self, dtype=None, **kwargs):
        super(Cast, self).__init__(**kwargs)
        self._dtype = dtype

    def call(self, inputs):
        return tf.cast(inputs, self._dtype)

    def get_config(self):
        config = super(Cast, self).get_config()
        config.update({"dtype": self._dtype})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, value):
        self._dtype = value 