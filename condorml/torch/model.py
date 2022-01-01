from typing import Optional, Dict, Any

import torch
from torch import nn


def concat_layers(inputs, layers, split_str="__SPLIT__"):

    dense_layer = []
    for input_col, pp in layers.items():
        if split_str in input_col:
            dense_layer.append(pp([inputs[col] for col in input_col.split(split_str)]))
        else:
            dense_layer.append(pp(inputs[input_col]))

    return torch.cat(dense_layer, dim=1)


def get_preprocess_layers_and_dims(pp_layers, split_str="__SPLIT__"):
    all_groups = {}
    for group, layers in pp_layers.items():
        all_groups[group] = {split_str.join(k) if isinstance(k, list) else k: v for k, v in layers}

    input_dim = {k: sum([l.output_dim() for _, l in layers]) for k, layers in pp_layers.items()}
    return all_groups, input_dim


class BaseTorchModel(nn.Module):
    def __init__(self, preprocess_layers, config: Optional[Dict[str, Any]] = None, hvd=None, *args, **kwargs):
        super().__init__()
        self.preprocess_layers, self.input_dimensions = get_preprocess_layers_and_dims(preprocess_layers)
        self.config = config
        self.hvd = hvd