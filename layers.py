import tensorflow as tf
from tensorflow.keras.layers import Layer

class Cast(Layer):
    def __init__(self, dtype, **kwargs):
        super(Cast, self).__init__(**kwargs)
        self.dtype = dtype

    def call(self, inputs):
        return tf.cast(inputs, self.dtype)

    def get_config(self):
        config = super(Cast, self).get_config()
        config.update({"dtype": self.dtype})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @property
    def dtype(self):
        return self.dtype

    @dtype.setter
    def dtype(self, value):
        self.dtype = value 