import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable(package="zenithml")
class WeightedNHotEncodingLayer(layers.Layer):
    def __init__(self, input_col: str, weight_col: str, num_buckets: int, combiner: str = "sum", **kwargs):
        name = f"{input_col}_weight_{weight_col}"
        super().__init__(name=name, dtype=tf.float32, **kwargs)
        self.combiner = combiner
        self.num_buckets = num_buckets
        self.trainable = False
        self.embedding_table = None

    def build(self, input_shapes):

        self.embedding_table = self.add_weight(
            name="{}/embedding_weights".format(self.name),
            trainable=False,
            initializer=tf.constant_initializer(np.eye(self.num_buckets)),
            shape=(self.num_buckets, self.num_buckets),
        )
        self.built = True

    def call(self, inputs, *args, **kwargs):
        inputs, weights = inputs

        if isinstance(inputs, tf.sparse.SparseTensor) or isinstance(inputs, list) or isinstance(inputs, tuple):
            values = inputs[0][:, 0]
            row_lengths = inputs[1][:, 0]

            weights_values = weights[0][:, 0]
            weights_row_lengths = weights[1][:, 0]

            inputs = tf.RaggedTensor.from_row_lengths(values, row_lengths).to_sparse()
            weights = tf.RaggedTensor.from_row_lengths(weights_values, weights_row_lengths).to_sparse()
            embeddings = tf.nn.safe_embedding_lookup_sparse(
                self.embedding_table, sparse_ids=inputs, sparse_weights=weights, combiner=self.combiner
            )
        else:
            raise Exception("Input must be sparse tensor for WeightedHotEncodingLayer")

        return embeddings
