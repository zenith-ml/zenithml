import numpy as np
import tensorflow as tf
from tensorflow.python.ops import array_ops

from condorml.tf.layers.preprocess.numerical import NumericalLayer


@tf.keras.utils.register_keras_serializable(package="condorml")
class MinMaxNormalizeLayer(NumericalLayer):
    def __init__(
        self,
        name: str,
        min_val: tf.float32 = None,
        max_val: tf.float32 = None,
        dtype: tf.dtypes = tf.float32,
        axis=-1,
        **kwargs,
    ):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.axis = self._get_axis(axis)
        self.min_val = min_val
        self.max_val = max_val

    def build(self, input_shape):
        input_shape = self._init_build(input_shape)
        # Create variables without keeping reduced axes.
        max_and_min_shape = tuple(input_shape[d] for d in self._keep_axis)

        self.max_val_weight = self.add_weight(
            name="max",
            shape=max_and_min_shape,
            dtype=self.dtype,
            initializer=tf.compat.v1.ones_initializer,
            trainable=False,
        )
        self.min_val_weight = self.add_weight(
            name="min",
            shape=max_and_min_shape,
            dtype=self.dtype,
            initializer=tf.compat.v1.zeros_initializer,
            trainable=False,
        )

        if self.max_val is not None or self.min_val is not None:
            max_val = self.max_val * np.ones(max_and_min_shape)
            min_val = self.min_val * np.ones(max_and_min_shape)
            self.set_weights([max_val, min_val])
        self.built = True

    def call(self, inputs, **kwargs):
        inputs = self._standardize_inputs(inputs)
        inputs = tf.cast(inputs, dtype=tf.float32)
        max_val = tf.cast(self.max_val_weight, dtype=tf.float32)
        min_val = tf.cast(self.min_val_weight, dtype=tf.float32)
        # We need to reshape the min and max data to ensure that Tensorflow broadcasts the data correctly.
        max_val = array_ops.reshape(max_val, self._broadcast_shape)
        min_val = array_ops.reshape(min_val, self._broadcast_shape)
        return (inputs - min_val) / (max_val - min_val)

    def get_config(self):
        config = super(MinMaxNormalizeLayer, self).get_config()
        config.update({"max_val": self.max_val.numpy()})
        config.update({"min_val": self.min_val.numpy()})
        return config


@tf.keras.utils.register_keras_serializable(package="condorml")
class LogNormalizeLayer(NumericalLayer):
    def __init__(self, name: str, log_threshold: tf.float32 = None, axis=-1, dtype: tf.dtypes = tf.float32, **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.axis = self._get_axis(axis)
        self.log_threshold = log_threshold

    def build(self, input_shape):
        input_shape = self._init_build(input_shape)

        # Create variables without keeping reduced axes.
        log_threshold_shape = tuple(input_shape[d] for d in self._keep_axis)

        self.log_threshold_weight = self.add_weight(
            name="log_threshold",
            shape=log_threshold_shape,
            dtype=self.dtype,
            initializer=tf.compat.v1.zeros_initializer,
            trainable=False,
        )

        super(LogNormalizeLayer, self).build(input_shape)

        if self.log_threshold is not None:
            log_threshold = self.log_threshold * np.ones(log_threshold_shape)
            self.set_weights([log_threshold])

        self.built = True

    def call(self, inputs, **kwargs):
        @tf.function
        def log10(x):
            numerator = tf.math.log(x)
            denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
            return numerator / denominator

        inputs = self._standardize_inputs(inputs)
        inputs = tf.cast(inputs, dtype=tf.float32)
        # We need to reshape the min and max data to ensure that Tensorflow broadcasts the data correctly.
        _log_threshold = array_ops.reshape(self.log_threshold_weight, self._broadcast_shape)
        return log10(inputs + _log_threshold)

    def get_config(self):
        config = super(LogNormalizeLayer, self).get_config()
        config.update({"log_threshold": self.log_threshold.numpy()})
        return config


@tf.keras.utils.register_keras_serializable(package="condorml")
class BucketizedLayer(NumericalLayer):
    pass
