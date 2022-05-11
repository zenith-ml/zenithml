import torch
import torch.nn as nn


class WeightedNHotEncodingLayer(nn.Module):
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
        x, weights = x

        if isinstance(x, list) or isinstance(x, tuple):
            values, offsets = x
            values = torch.squeeze(values, -1)

            weights_values, weights_offsets = weights
            weights_values = torch.squeeze(weights_values, -1)
            return self.embedding_table(values, offsets[:, 0], per_sample_weights=weights_values)
        else:
            raise Exception("Input must be list or tuple for WeightedHotEncodingLayer")

    def output_dim(self):
        return self.num_buckets
