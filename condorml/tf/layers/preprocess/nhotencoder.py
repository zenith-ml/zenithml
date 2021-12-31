import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable(package="condorml")
class NHotEncodingLayer(layers.Layer):
    def __init__(
        self,
        name: str,
        num_buckets: int,
        combiner: str = "sum",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
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
        inputs = inputs

        if isinstance(inputs, tf.sparse.SparseTensor) or isinstance(inputs, list) or isinstance(inputs, tuple):
            values = inputs[0][:, 0]
            row_lengths = inputs[1][:, 0]
            inputs = tf.RaggedTensor.from_row_lengths(values, row_lengths).to_sparse()
            embeddings = tf.nn.safe_embedding_lookup_sparse(
                self.embedding_table, sparse_ids=inputs, combiner=self.combiner
            )
        else:
            embeddings = tf.gather(self.embedding_table, inputs[:, 0])

        return embeddings

    def get_config(self):
        config = super(NHotEncodingLayer, self).get_config()
        config.update({"num_buckets": self.num_buckets})
        config.update({"combiner": self.combiner})
        return config
