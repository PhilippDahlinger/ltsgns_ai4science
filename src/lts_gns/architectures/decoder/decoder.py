import torch.nn
from torch import nn


class Decoder(torch.nn.Module):
    """
    The decoder of the "encode-process-decode" structure of the GNS. Consists of a decode_module that takes
    arbitrary inputs, and a readout_module that takes the output of the decode_module and produces the final output.
    """

    def __init__(self, decode_module: nn.Module, readout_module: nn.Module, device: str | torch.device):
        super().__init__()
        self._device = device
        self._decode_module = decode_module.to(device)
        self._readout_module = readout_module.to(device)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """

        Args:


        Returns:

        """
        return self._readout_module(self._decode_module(*args, **kwargs))

    def __repr__(self):
        return f"Decoder(decode_module={self._decode_module}, readout_module={self._readout_module})"
