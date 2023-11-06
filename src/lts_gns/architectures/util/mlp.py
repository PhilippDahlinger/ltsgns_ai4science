from torch import nn as nn
import torch
from torch.nn import utils
import numpy as np

from lts_gns.util.config_dict import ConfigDict
from lts_gns.util.own_types import *


class SaveBatchnorm1d(nn.BatchNorm1d):
    # todo get rid of the if/else here?
    """
    Custom Layer to deal with Batchnorms for 1-sample inputs
    """

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Normalization layer
        Args:
            tensor: The input graph

        Returns: The normalized graph

        """
        batch_size = tensor.shape[0]
        if batch_size == 1 or len(tensor.shape) == 1:
            return tensor
        else:
            return super().forward(input=tensor)


def build_mlp(*,
              in_features: int,
              config: ConfigDict,
              latent_dimension: int,
              device: Optional[str] = None) -> nn.ModuleList:
    """
    Builds a latent MLP, i.e., a network that gets as input a number of latent features, and then feeds these features
     through a number of fully connected layers and activation functions.
    Args:
        in_features: Number of input features
        config: Dictionary containing the specification for the mlp network.
          Includes
          * num_layers: how many layers+activations to build
          * activation_function: which activation function to use
          * add_output_layer: whether to add an additional output layer
          * regularization: A subdictionary containing
            * spectral_norm: Whether to include spectral normalization
            * dropout: Dropout rate for the activations. Applied *after* the activation function
            * latent_normalization: Whether to use "batch" or "layer" normalization, or no normalization at all
        latent_dimension: Latent size of the mlp layers.
        device:  Device to use for the module. If None, the default device is used.
    Returns: A nn.ModuleList representing the MLP module

    """
    mlp_layers = nn.ModuleList()

    activation_function: str = config.activation_function.lower()
    add_output_layer = config.add_output_layer
    network_layout = np.repeat(latent_dimension, config.num_layers)
    regularization_config = config.regularization

    spectral_norm: bool = regularization_config.get("spectral_norm", False)
    dropout = regularization_config.get("dropout", 0.0)
    latent_normalization = regularization_config.get("latent_normalization", None)

    previous_shape = in_features

    # build actual network
    for current_layer_num, current_layer_size in enumerate(network_layout):
        # add main linear layer
        if spectral_norm:
            mlp_layers.append(utils.spectral_norm(nn.Linear(in_features=previous_shape,
                                                            out_features=current_layer_size)))
        else:
            mlp_layers.append(nn.Linear(in_features=previous_shape,
                                        out_features=current_layer_size))

        # activation function
        if activation_function == "relu":
            mlp_layers.append(nn.ReLU())
        elif activation_function == "leakyrelu":
            mlp_layers.append(nn.LeakyReLU())
        elif activation_function == "elu":
            mlp_layers.append(nn.ELU())
        elif activation_function in ["swish", "silu"]:
            mlp_layers.append(nn.SiLU())
        elif activation_function == "mish":
            mlp_layers.append(nn.Mish())
        elif activation_function == "gelu":
            mlp_layers.append(nn.GELU())
        elif activation_function == "tanh":
            mlp_layers.append(nn.Tanh())
        else:
            raise ValueError("Unknown activation function '{}'".format(activation_function))

        # add latent normalization method
        if latent_normalization in ["batch", "batch_norm"]:
            mlp_layers.append(SaveBatchnorm1d(num_features=latent_dimension))
        elif latent_normalization in ["layer", "layer_norm"]:
            mlp_layers.append(nn.LayerNorm(normalized_shape=latent_dimension))

        # add dropout
        if dropout:
            mlp_layers.append(nn.Dropout(p=dropout))

        previous_shape = current_layer_size

    if add_output_layer is not None and add_output_layer:  # add another layer without activation for the output
        mlp_layers.append(nn.Linear(previous_shape, latent_dimension))
    if device is not None:
        mlp_layers.to(device)
    return mlp_layers


class MLP(nn.Module):
    """
    Feedforward module. Gets some input x and computes an output f(x).
    """

    def __init__(self, *, in_features: int,
                 latent_dimension: int,
                 config: ConfigDict,
                 device: Optional[str] = None):
        """
        Initializes a new encoder module that consists of multiple different layers that together encode some
        input into a latent space
        Args:
            in_features: The input shape for the feedforward network
            latent_dimension: Latent size of the mlp layers.
            config: Dict containing information about how to build and regularize the MLP (via batchnorm etc.)
            device:  Device to use for the module. If None, the default device is used.
        """
        super().__init__()
        self.mlp_layers = build_mlp(in_features=in_features,
                                    config=config,
                                    latent_dimension=latent_dimension,
                                    device=device)
        self.in_features = in_features

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Computes the forward pass for the given input tensor
        Args:
            tensor: Some input tensor x

        Returns: The processed tensor f(x)

        """
        for mlp_layer in self.mlp_layers:
            tensor = mlp_layer(tensor)
        return tensor
