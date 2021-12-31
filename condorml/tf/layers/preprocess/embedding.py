import tensorflow as tf
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable(package="condorml")
class EmbeddingLayer(layers.Layer):
    def __init__(
        self,
        name: str,
        dimensions: int,
        num_buckets: int,
        combiner: str = "sum",
        seq_length: int = 1,
        weights=None,
        **kwargs,
    ):

        super().__init__(name=name, **kwargs)
        self.combiner = combiner
        self.num_buckets = num_buckets
        self._weights = weights
        self.dimensions = dimensions
        self.seq_length = seq_length
        self.trainable = False
        self.embedding_table = None

    def build(self, input_shapes):
        self.embedding_table = self.add_weight(
            name="{}/embedding_weights".format(self.name),
            trainable=True,
            initializer="glorot_normal",
            shape=(self.num_buckets, self.dimensions),
        )
        if self._weights is not None:
            tf.compat.v1.assign(
                self.embedding_table,
                tf.cast(tf.experimental.dlpack.from_dlpack(self._weights.toDlpack()), dtype=tf.float32),
            )

        self.built = True

    def call(self, inputs, *args, **kwargs):
        # if not self._nvt:
        #     x = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=self.vocab, mask_token=None)(x)

        if isinstance(inputs, tf.sparse.SparseTensor) or isinstance(inputs, list) or isinstance(inputs, tuple):
            values = inputs[0][:, 0]
            row_lengths = inputs[1][:, 0]
            inputs = tf.RaggedTensor.from_row_lengths(values, row_lengths).to_sparse()
            if self.seq_length > 1:
                embeddings = tf.gather(self.embedding_table, tf.sparse.to_dense(inputs))
            else:
                embeddings = tf.nn.safe_embedding_lookup_sparse(
                    self.embedding_table, sparse_ids=inputs, combiner=self.combiner
                )

        elif self.seq_length > 1:
            embeddings = tf.gather(self.embedding_table, inputs)
        else:
            embeddings = tf.gather(self.embedding_table, inputs[:, 0])

        return embeddings

    def get_config(self):
        config = super(EmbeddingLayer, self).get_config()
        config.update({"num_buckets": self.num_buckets})
        config.update({"combiner": self.combiner})
        config.update({"dimensions": self.dimensions})
        config.update({"seq_length": self.seq_length})
        return config
