from typing import List, Dict

from torch import nn
import torch

from lts_gns.architectures.util.mlp import MLP
from lts_gns.architectures.util.fourier_embedding import FourierEmbedding
from lts_gns.util import keys
from lts_gns.util.config_dict import ConfigDict


class LTSGNSDecodeModule(nn.Module):

    def __init__(self, decoder_config: ConfigDict, in_feature_dict: Dict[str, int], device: str):
        """
        Decoder module for the LTSGNS architecture.
        Args:
            decoder_config:
            in_feature_dict:
            device:
        """
        super().__init__()
        if decoder_config.z_embedding == "linear":
            num_in_features = in_feature_dict[keys.PROCESSOR_DIMENSION] + decoder_config.latent_dimension
            self._z_embedding = nn.Linear(in_features=in_feature_dict[keys.Z_DIMENSION],
                                          out_features=decoder_config.latent_dimension,
                                          device=device)
        elif decoder_config.z_embedding == "fourier":
            num_in_features = in_feature_dict[keys.PROCESSOR_DIMENSION] + 2 * decoder_config.latent_dimension
            self._z_embedding = FourierEmbedding(in_features=in_feature_dict[keys.Z_DIMENSION],
                                                 half_out_features=decoder_config.latent_dimension,
                                                 sigma=1.0,
                                                 device=device)
        elif decoder_config.z_embedding is None:
            num_in_features = sum([value for value in in_feature_dict.values()])
            self._z_embedding = lambda x: x
        else:
            raise ValueError(f"Unknown z_embedding {decoder_config.z_embedding}")

        self._mlp = MLP(in_features=num_in_features,
                        latent_dimension=decoder_config.latent_dimension,
                        config=ConfigDict(activation_function="relu",
                                          add_output_layer=False,
                                          num_layers=1,
                                          regularization={},
                                          ),
                        device=device
                        )
        if decoder_config.feature_dropout is None:
            self.feature_dropout = lambda x: x
        else:
            self.feature_dropout = nn.Dropout(p=decoder_config.feature_dropout)

    def forward(self, processor_output: torch.Tensor, z: torch.Tensor,
                mesh_nodes_per_task: List[int]) -> torch.Tensor:
        """
        Forward pass of the decoder.
        Args:
            processor_output:
            z:
            mesh_nodes_per_task:

        Returns:

        """
        processor_output = self.feature_dropout(processor_output)
        processor_output = processor_output.repeat(z.shape[0], 1, 1)
        z = self._z_embedding(z)  # potentially embed z to the hidden dimension
        reshaped_z = torch.repeat_interleave(z, mesh_nodes_per_task, dim=1)
        # reshaped_z has now shape (num_samples, n_vertices, d_z)
        # gnn_output has shape (1, n_vertices, d_output)

        combined_input = torch.cat([processor_output, reshaped_z], dim=2)
        # combined_output has shape (num_samples, n_vertices, d_output + d_z)
        return self._mlp(combined_input)
