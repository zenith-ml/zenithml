import torch
import torch.nn as nn

from zenithml.torch.layers import NumericalLayer


class NormalizationLayer(nn.Module):
    EPS = torch.finfo(torch.float32).eps

    def __init__(self, mean: float, variance: float, dimension: int = 1):
        super().__init__()
        self.mean = nn.Parameter(torch.tensor(mean, dtype=torch.float32, requires_grad=False))
        self.variance = nn.Parameter(torch.tensor(variance, dtype=torch.float32, requires_grad=False))
        self.dimension = dimension

    def forward(self, x: torch.Tensor):
        x = x.float().view(-1, self.dimension)
        mean = self.mean
        variance = self.variance

        return (x - mean) / torch.maximum(torch.sqrt(variance), torch.tensor(self.EPS))

    def output_dim(self):
        return self.dimension


class MinMaxNormalizeLayer(nn.Module):
    def __init__(self, min_val: float, max_val: float, dimension: int = 1):
        super().__init__()

        self.min_val = nn.Parameter(torch.tensor(min_val, dtype=torch.float32, requires_grad=False))
        self.max_val = nn.Parameter(torch.tensor(max_val, dtype=torch.float32, requires_grad=False))
        self.dimension = dimension

    def forward(self, x: torch.Tensor):
        x = x.float().view(-1, self.dimension)
        max_val = self.max_val
        min_val = self.min_val
        return (x - min_val) / (max_val - min_val)

    def output_dim(self):
        return self.dimension


class LogNormalizeLayer(nn.Module):
    def __init__(self, log_threshold: float, dimension: int = 1):
        super().__init__()
        self.log_threshold = nn.Parameter(torch.tensor(log_threshold, dtype=torch.float32, requires_grad=False))
        self.dimension = dimension

    def forward(self, x: torch.Tensor):
        x = x.float().view(-1, self.dimension)
        _log_threshold = self.log_threshold
        return torch.log10(x + _log_threshold)

    def output_dim(self):
        return self.dimension


class BucketizedLayer(NumericalLayer):
    pass
