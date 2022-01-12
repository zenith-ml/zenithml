import tensorflow as tf
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable(package="zenithml")
class NumericalLayer(layers.Layer):
    def __init__(self, name: str, dimension: int = 1, dtype=tf.float32, **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.dimension = dimension
        self.trainable = False

    def compute_output_shape(self, input_shape):
        if self.dimension > 1:
            return input_shape, self.dimension
        else:
            return super().compute_output_shape(input_shape)

    def _get_axis(self, axis):
        # Standardize `axis` to a tuple.
        if axis is None:
            axis = ()
        elif isinstance(axis, int):
            axis = (axis,)
        else:
            axis = tuple(axis)
        if 0 in axis:
            raise ValueError("The argument 'axis' may not be 0.")
        return axis

    def _standardize_inputs(self, inputs):
        inputs = tf.convert_to_tensor(inputs)
        if inputs.shape.rank == 0:
            inputs = tf.reshape(inputs, [1, 1])
        elif inputs.shape.rank == 1:
            inputs = tf.compat.v1.expand_dims(inputs, 1)

        if inputs.dtype != self.dtype:
            inputs = tf.cast(inputs, self.dtype)
        return inputs

    def _init_build(self, input_shape):
        input_shape = input_shape
        input_shape = tf.TensorShape(input_shape).as_list()
        if len(input_shape) == 1:
            input_shape = input_shape + [1]
        ndim = len(input_shape)
        if any(a < 1 - ndim or a >= ndim for a in self.axis):
            raise ValueError(
                "All `axis` values must be in the range "
                "[1 - ndim, ndim - 1]. Found "
                "ndim: `{}`, axis: {}".format(ndim, self.axis)
            )
        # Axes to be kept, replacing negative values with positive equivalents.
        # Sorted to avoid transposing axes.
        self._keep_axis = sorted([d if d >= 0 else d + ndim for d in self.axis])
        # Axes to be reduced.
        self._reduce_axis = [d for d in range(ndim) if d not in self._keep_axis]
        # 1 if an axis should be reduced, 0 otherwise.
        self._reduce_axis_mask = [0 if d in self._keep_axis else 1 for d in range(ndim)]
        # Broadcast any reduced axes.
        self._broadcast_shape = [input_shape[d] if d in self._keep_axis else 1 for d in range(ndim)]
        return input_shape

    def call(self, inputs, *args, **kwargs):
        inputs = inputs
        if isinstance(inputs, tf.sparse.SparseTensor) or isinstance(inputs, list) or isinstance(inputs, tuple):
            values, row_lengths = inputs[0][:, 0], inputs[1][:, 0]
            inputs = tf.RaggedTensor.from_row_lengths(values, row_lengths, name=self.name + "_ragged").to_tensor()
            # inputs = tf.sparse.to_dense(inputs)
            inputs.set_shape(self.compute_output_shape(inputs.shape[0]))
        return tf.cast(inputs, dtype=tf.float32)

    def get_config(self):
        config = super(NumericalLayer, self).get_config()
        config.update({"dimension": self.dimension})
        return config
