from typing import List

import torch

from lts_gns.architectures.normalizers.normalizer_util import identity


class WorldToModelNormalizer(torch.nn.Module):
    """
    Normalizes the desired outputs of the model to have zero mean and unit variance. I.e., moves from "world space"
    in terms of model outputs to "model space", which is more convenient for training.
    """

    def __init__(self, use_output_normalizer: bool, world_mean: torch.Tensor | List, world_std: torch.Tensor | List):
        """

        Args:
            use_output_normalizer: Whether to use the normalizer or not. If False, will use an identity function instead
            world_mean:
            world_std:
        """
        super().__init__()
        if use_output_normalizer:
            self._maybe_normalize = self._normalize
        else:
            self._maybe_normalize = identity
        self.world_mean = torch.nn.Parameter(torch.tensor(world_mean), requires_grad=False)
        self.world_std = torch.nn.Parameter(torch.tensor(world_std), requires_grad=False)

    def _normalize(self, x):
        return (x - self.world_mean) / self.world_std

    def forward(self, x):
        return self._maybe_normalize(x)
