import torch
from torch import nn


class NoScaleDropout(nn.Module):
    """
        Dropout without rescaling.
    """

    def __init__(self, p: float) -> None:
        super().__init__()
        self.rate = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.rate == 0:
            return x
        else:
            mask = torch.empty(x.shape, device=x.device).bernoulli_(1 - self.rate)
            return x * mask


class NoScaleMultiRateDropout(nn.Module):
    """
        Dropout without rescaling and variable dropout rates.
    """

    def __init__(self, p_max) -> None:
        super().__init__()
        self.rate_max = p_max

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.rate_max == 0:
            return x
        else:
            rate = torch.empty(1, device=x.device).uniform_(0, self.rate_max)
            mask = torch.empty(x.shape, device=x.device).bernoulli_(1 - rate)
            return x * mask