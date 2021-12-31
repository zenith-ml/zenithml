import torch
import torch.nn as nn
from nvtabular.dispatch import HAS_GPU


class NumericalLayer(nn.Module):
    def __init__(self, dimension: int = 1):
        super().__init__()
        self.dimension = dimension

    @property
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _pull_values_offsets(self, values_offset):
        # pull_values_offsets, return values offsets diff_offsets
        if isinstance(values_offset, tuple):
            values = values_offset[0].flatten()
            offsets = values_offset[1].flatten()
        else:
            values = values_offset.flatten()
            offsets = torch.arange(values.size()[0], device=self.device)
        num_rows = len(offsets)
        if HAS_GPU:
            offsets = torch.cat([offsets, torch.cuda.LongTensor([len(values)])])
        else:
            offsets = torch.cat([offsets, torch.LongTensor([len(values)])])
        diff_offsets = offsets[1:] - offsets[:-1]
        return values, offsets, diff_offsets, num_rows

    def _get_indices(self, offsets, diff_offsets):
        row_ids = torch.arange(len(offsets) - 1, device=self.device)
        row_ids_repeated = torch.repeat_interleave(row_ids, diff_offsets)
        row_offset_repeated = torch.repeat_interleave(offsets[:-1], diff_offsets)
        col_ids = torch.arange(len(row_offset_repeated), device=self.device) - row_offset_repeated
        indices = torch.cat([row_ids_repeated.unsqueeze(-1), col_ids.unsqueeze(-1)], axis=1)
        return indices

    def forward(self, x: torch.Tensor):
        if isinstance(x, tuple) or isinstance(x, list):
            values, offsets, diff_offsets, num_rows = self._pull_values_offsets(x)
            indices = self._get_indices(offsets, diff_offsets)
            return torch.sparse_coo_tensor(
                indices.T, values, torch.Size([num_rows, self.dimension]), device=self.device
            ).to_dense()
            # value, _ = x
            # return value.float().view(-1, self.dimension)

        return x.float().view(-1, self.dimension)

    def output_dim(self):
        return self.dimension
