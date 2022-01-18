from pathlib import Path
from typing import Optional, Dict, Any, Union

import torch
from nvtabular import dispatch
from torch import nn

from zenithml.utils import rich_logging, fs


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

    def loss_fn(self, logits, truth):
        raise NotImplementedError

    @staticmethod
    def trainer(model, ds, optimizer, config, model_dir: Optional[Union[str, Path]] = None):

        logger = rich_logging()
        if dispatch.HAS_GPU:
            model = model.to("cuda")

        for epoch in range(config.get("num_epoch")):
            for idx, batch in enumerate(ds):
                x, y = batch
                logits = model.forward(x)
                loss = model.loss_fn(logits, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if idx % 10 == 0:
                    logger.info(f"\tBatch {idx:02d}. Train loss: {loss:.4f}.")
            logger.info(f"Epoch {epoch:02d}. Train loss: {loss:.4f}.")

        if model_dir:
            fs.mkdir(model_dir)
            model_path = fs.join(model_dir, "model.pt")
            logger.info(f"Saving Model to {model_path}")
            torch.save(model, f=fs.open_fileptr(model_path, mode="wb"))

    @staticmethod
    def load(model_dir: Union[str, Path]):
        model_path = fs.join(model_dir, "model.pt")
        assert model_path, f"{model_path} does not exist."
        return torch.load(f=fs.open_fileptr(model_path, mode="rb"))
