from typing import List

import torch

from lts_gns.architectures.normalizers.normalizer_util import identity


class ModelToWorldNormalizer(torch.nn.Module):
    def __init__(self, use_output_normalizer: bool, world_mean: torch.Tensor | List, world_std: torch.Tensor | List):
        """
        De-Normalizes the model outputs to have the same mean and variance as the world outputs.
        Args:
            use_output_normalizer: Whether to use the normalizer or not. If False, will use an identity function instead
            world_mean:
            world_std:
        """
        super().__init__()
        if use_output_normalizer:
            self._maybe_denormalize = self._denormalize
        else:
            self._maybe_denormalize = identity
        self.world_mean = torch.nn.Parameter(torch.tensor(world_mean), requires_grad=False)
        self.world_std = torch.nn.Parameter(torch.tensor(world_std), requires_grad=False)

    def _denormalize(self, x):
        return x * self.world_std + self.world_mean

    def forward(self, x):
        return self._maybe_denormalize(x)
