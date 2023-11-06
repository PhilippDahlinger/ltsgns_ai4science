from typing import Dict

import torch
from lts_gns.architectures.decoder.decoder import Decoder
from lts_gns.util.config_dict import ConfigDict


class DecoderFactory:
    def __call__(self, *args, **kwargs):
        return self.build_decoder(*args, **kwargs)

    @staticmethod
    def build_decoder(decoder_config: ConfigDict, action_dim: int, device: str, input_dimensions: Dict[str, int],
                      simulator_class: str):
        """
        Build a decoder, which consists of a decode_module and a readout_module.
        Args:
            decoder_config:
            action_dim:
            device:
            input_dimensions: Dictionary containing the dimensions of the input features

        Returns:

        """
        # TODO: different activation functions
        # TODO: multiple layers option
        if simulator_class == "MGNSimulator" or simulator_class == "MGNProDMPSimulator":
            from lts_gns.architectures.util.mlp import MLP
            decode_module = MLP(in_features=sum([value for value in input_dimensions.values()]),  # concat
                                latent_dimension=decoder_config.latent_dimension,
                                config=ConfigDict(activation_function="relu",
                                                  add_output_layer=False,
                                                  num_layers=1,
                                                  regularization={},
                                                  ),
                                device=device
                                )
        elif simulator_class == "LTSGNSSimulator" or simulator_class == "LTSGNSProDMPSimulator":
            from lts_gns.architectures.decoder.ltsgns_decode_module import LTSGNSDecodeModule
            decode_module = LTSGNSDecodeModule(decoder_config=decoder_config,
                                               in_feature_dict=input_dimensions,
                                               device=device)
        elif simulator_class == "CNPSimulator":
            from lts_gns.architectures.decoder.cnp_decode_module import CNPDecodeModule
            decode_module = CNPDecodeModule(decoder_config=decoder_config,
                                            in_feature_dict=input_dimensions,
                                            device=device)

        else:
            raise NotImplementedError(f"Decoder for simulator class {simulator_class} not implemented")

        readout_module = torch.nn.Linear(decoder_config.latent_dimension, action_dim)
        decoder = Decoder(decode_module, readout_module, device=device)
        return decoder
