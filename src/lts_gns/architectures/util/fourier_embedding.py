import einops
import numpy as np
import torch
from torch import nn

from lts_gns.util.config_dict import ConfigDict


class FourierEmbedding(nn.Module):
    """
    Implements the Fourier Embedding as described in the paper.
    """

    def __init__(self, in_features: int, half_out_features: int, sigma: float, device: str):
        """
        Initializes the Fourier Embedding.
        Args:
            in_features: The dimension of the input.
            half_out_features: Half of the dimension of the output. (Output is doubled due to sin and cos part)
            sigma: The standard deviation of the Gaussian distribution from which the frequencies are sampled.
        """
        super().__init__()
        self.device = device
        self.freqs = torch.randn(half_out_features, in_features) * sigma
        self.freqs = self.freqs.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the Fourier Embedding to the input.
        Args:
            x: The input to which the Fourier Embedding should be applied.

        Returns:
            The input with the Fourier Embedding applied.
        """
        linear_part = 2 * np.pi * einops.einsum(self.freqs, x, "output_dim input_dim, ... input_dim -> ... output_dim")
        return torch.cat([torch.sin(linear_part), torch.cos(linear_part)], dim=-1)
