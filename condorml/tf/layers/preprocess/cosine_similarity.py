import tensorflow as tf
from tensorflow.keras import layers

from condorml.tf.layers.preprocess.numerical import NumericalLayer


@tf.keras.utils.register_keras_serializable(package="condorml")
class CosineSimilarityLayer(layers.Layer):
    def __init__(self, var1_name: str, var2_name: str, dimension: int, **kwargs):
        name = f"{var1_name}_cosine_{var2_name}"
        self.dimension = dimension
        self.numeric_layer1 = NumericalLayer(name=var1_name, dimension=dimension)
        self.numeric_layer2 = NumericalLayer(name=var2_name, dimension=dimension)
        super().__init__(name=name, dtype=tf.float32, **kwargs)

        self.trainable = False

    def call(self, inputs, *args, **kwargs):
        x, y = inputs
        x, y = self.numeric_layer1(x), self.numeric_layer2(y)
        normalize_a = tf.nn.l2_normalize(x, axis=1)
        normalize_b = tf.nn.l2_normalize(y, axis=1)
        return tf.reshape(tf.reduce_sum(tf.multiply(normalize_a, normalize_b), axis=1), (-1, 1))
