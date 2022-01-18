import torch
import torch.nn as nn


class NHotEncodingLayer(nn.Module):
    def __init__(self, num_buckets: int, mode: str = "sum"):
        super().__init__()
        self.mode = mode
        self.num_buckets = num_buckets
        self.embedding_table = nn.EmbeddingBag.from_pretrained(
            torch.eye(num_buckets, num_buckets),
            mode=self.mode,
            freeze=True,
        )

    def forward(self, x):
        if isinstance(x, list) or isinstance(x, tuple):
            values, offsets = x
            values = torch.squeeze(values, -1)
            return self.embedding_table(values, offsets[:, 0])
        else:
            if len(x.shape) <= 1:
                x = x.unsqueeze(0)
            return self.embedding_table(x.view(-1, 1))

    def output_dim(self):
        return self.num_buckets
