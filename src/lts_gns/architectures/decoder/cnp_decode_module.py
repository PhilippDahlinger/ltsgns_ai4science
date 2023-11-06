from typing import List, Dict

from torch import nn
import torch

from lts_gns.architectures.util.mlp import MLP
from lts_gns.architectures.util.fourier_embedding import FourierEmbedding
from lts_gns.util import keys
from lts_gns.util.config_dict import ConfigDict


class CNPDecodeModule(nn.Module):

    def __init__(self, decoder_config: ConfigDict, in_feature_dict: Dict[str, int], device: str):
        """
        Decoder module for the LTSGNS architecture.
        Args:
            decoder_config:
            in_feature_dict:
            device:
        """
        super().__init__()

        num_in_features = sum([value for value in in_feature_dict.values()])
        self._mlp = MLP(in_features=num_in_features,
                        latent_dimension=decoder_config.latent_dimension,
                        config=ConfigDict(activation_function="relu",
                                          add_output_layer=False,
                                          num_layers=1,
                                          regularization={},
                                          ),
                        device=device
                        )

    def forward(self, processor_output: torch.Tensor, r: torch.Tensor,
                mesh_nodes_per_task: List[int]) -> torch.Tensor:
        """
        Forward pass of the decoder.
        Args:
            processor_output:
            r:
            mesh_nodes_per_task:

        Returns:

        """
        reshaped_r = torch.repeat_interleave(r, mesh_nodes_per_task, dim=0)
        # reshaped_z has now shape (n_vertices, d_z)
        # gnn_output has shape (n_vertices, d_output)

        combined_input = torch.cat([processor_output, reshaped_r], dim=1)
        # combined_input has shape (n_vertices, d_output + d_z)
        return self._mlp(combined_input)
