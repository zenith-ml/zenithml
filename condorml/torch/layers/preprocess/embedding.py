from typing import Union, Tuple

import torch
import torch.nn as nn
from nvtabular.dispatch import HAS_GPU


class EmbeddingLayer(nn.Module):
    def __init__(
        self,
        dimensions: int,
        num_buckets: int,
        mode: str = "sum",
        seq_length: int = 1,
        trainable=False,
        weights=None,
        use_gpu=True,
    ):

        super().__init__()

        self.num_buckets = num_buckets
        self._weights = weights
        self.dimensions = dimensions
        self.seq_length = nn.Parameter(torch.tensor(seq_length, dtype=torch.long), requires_grad=False)
        self.mode = mode

        if weights:
            self.embedding_table: Union[nn.EmbeddingBag, nn.Embedding] = nn.EmbeddingBag.from_pretrained(
                embeddings=torch.tensor(torch.utils.dlpack.from_dlpack(self._weights.toDlpack()), dtype=torch.float32),
                mode=self.mode,
                freeze=trainable,
            )
        else:
            assert trainable is False, "weights must be pass if trainable is False"
            if self.seq_length > 1:
                self.embedding_table = nn.Embedding(self.num_buckets, self.dimensions, padding_idx=0)
            else:

                self.embedding_table = nn.EmbeddingBag(self.num_buckets, self.dimensions, mode=mode, padding_idx=0)

    def forward(self, x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]):
        if self.seq_length == 1:
            if isinstance(x, list) or isinstance(x, tuple):
                values, offsets = x
                values = torch.squeeze(values, -1)
                return self.embedding_table(values, offsets[:, 0])
            else:
                if len(x.shape) <= 1:
                    x = x.unsqueeze(0)
                return self.embedding_table(x.view(-1, 1))
        elif isinstance(x, torch.Tensor):
            return self.embedding_table(x.view(-1, self.seq_length))  # type: ignore

    def output_dim(self):
        return self.dimensions
