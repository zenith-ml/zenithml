from abc import ABC
from typing import Dict, Any, Optional

import tensorflow as tf
from tensorflow.keras.layers import Layer


def concat_layers(inputs, layers):
    dense_layer = []
    for input_col, pp in layers:
        if isinstance(input_col, list):
            dense_layer.append(pp([inputs[col] for col in input_col]))
        else:
            dense_layer.append(pp(inputs[input_col]))
    return tf.keras.layers.Concatenate()(dense_layer)


class BaseKerasModel(tf.keras.Model, ABC):
    def __init__(self, preprocess_layers: Layer, config: Optional[Dict[str, Any]] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pp_layers = preprocess_layers
        self.config = config

    @property
    def preprocess_layers(self):
        return self._pp_layers
