from abc import ABC
from typing import Dict, Any, Optional

import tensorflow as tf


def concat_layers(inputs, layers, split_str="__SPLIT__"):

    dense_layer = []
    for input_col, pp in layers.items():
        if split_str in input_col:
            dense_layer.append(pp([inputs[col] for col in input_col.split(split_str)]))
        else:
            dense_layer.append(pp(inputs[input_col]))

    return tf.keras.layers.Concatenate()(dense_layer)


def get_preprocess_layers_and_dims(pp_layers, split_str="__SPLIT__"):
    all_groups = {}
    for group, layers in pp_layers.items():
        all_groups[group] = {split_str.join(k) if isinstance(k, list) else k: v for k, v in layers}

    return all_groups


class BaseKerasModel(tf.keras.Model, ABC):
    def __init__(self, preprocess_layers, config: Optional[Dict[str, Any]] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.preprocess_layers, self.input_dimensions = get_preprocess_layers_and_dims(preprocess_layers)
        self.config = config
