from typing import Dict

import torch
from hmpn import AbstractMessagePassingBase, get_hmpn_from_graph
from torch_geometric.data import Batch
from torch_scatter import scatter

from lts_gns.algorithms.simulators.simulator_util import unpack_node_features
from lts_gns.architectures.decoder.decoder import Decoder
from lts_gns.architectures.util.mlp import MLP
from lts_gns.util import keys
from lts_gns.util.config_dict import ConfigDict
from lts_gns.util.util import node_type_mask


class CNPEncoder(torch.nn.Module):
    def __init__(self, encoder_config, env, device):
        super().__init__()
        gnn_config = encoder_config.gnn
        self._config = encoder_config
        example_input_batch = env.get_next_train_task_batch(add_training_noise=False)[0]
        example_input_batch = self.preprocess_context_batch(example_input_batch)

        self._gnn: AbstractMessagePassingBase = get_hmpn_from_graph(example_graph=example_input_batch,
                                                                    latent_dimension=gnn_config.latent_dimension,
                                                                    node_name=keys.MESH,
                                                                    unpack_output=False,  # return full graph
                                                                    base_config=gnn_config.base,
                                                                    device=device)

        first_output_layer = MLP(in_features=gnn_config.latent_dimension,
                                 latent_dimension=gnn_config.latent_dimension,
                                 config=ConfigDict(activation_function="relu",
                                                   add_output_layer=False,
                                                   num_layers=1,
                                                   regularization={},
                                                   ),
                                 device=device
                                 )
        second_output_layer = torch.nn.Linear(gnn_config.latent_dimension, self._config.d_r)
        self._output_layer = Decoder(first_output_layer, second_output_layer, device=device)

    def forward(self, context_batch, task_belonging):
        context_batch = self.preprocess_context_batch(context_batch)
        out_graph = self._gnn(context_batch)
        node_features = unpack_node_features(out_graph, keys.MESH)  # unpack the node feature
        r_n = self._output_layer(node_features)
        return self.aggregate_r_n(r_n, task_belonging)

    def preprocess_context_batch(self, batch: Batch) -> Batch:
        """
        Adds the y label as part of the x input to combine both in order to get the r_n with the gnn.
        Args:
            batch:

        Returns: new batch with y in x
        """
        # copy a batch to not change the original batch
        batch = batch.clone()
        mesh_node_idx = node_type_mask(batch, key=keys.MESH)
        expanded_y = torch.zeros((batch.x.shape[0], batch.y.shape[1]), dtype=batch.x.dtype, device=batch.x.device)
        expanded_y[mesh_node_idx] = batch.y
        # add the expanded y to the x input
        batch.x = torch.cat([batch.x, expanded_y], dim=1)
        return batch

    def aggregate_r_n(self, r_n, task_belonging) -> torch.Tensor:
        """
        Use torch scatter to mean the timesteps of the corresponding subtasks
        Args:
            r_n:
            task_belonging:

        Returns:

        """
        # assumes that the mesh nodes per time step in one batch is fixed
        mesh_nodes_per_time_step = task_belonging[keys.MESH_NODES_PER_TIME][0]
        r_n = r_n.reshape((-1, mesh_nodes_per_time_step, self._config.d_r))
        # mean over every node in the mesh
        r_n = torch.mean(r_n, dim=1)
        # scatter over the time steps
        index = task_belonging[keys.TIME_STEPS_PER_SUBTASK]
        # index == (2, 3) I want it to be (0, 0, 1, 1, 1)
        index = torch.repeat_interleave(torch.arange(len(index), device=r_n.device), torch.tensor(index, device=r_n.device))


        r = scatter(src=r_n, index=index, dim=0, reduce="mean")
        return r

    def save_checkpoint(self, directory: str, iteration: int, is_initial_save: bool, is_final_save: bool = False):
        """
        Saves the state dict of the simulator to the specified directory.
        Args:
            directory:
            iteration:
            is_final_save:

        Returns:

        """
        gnn_params = self._gnn.state_dict()
        output_layer_params = self._output_layer.state_dict()
        save_dict = {"gnn_params": gnn_params,
                     "output_layer_params": output_layer_params}
        if is_final_save:
            file_name = f"cnp_encoder_state_dict_final.pt"
        else:
            file_name = f"cnp_encoder_state_dict_{iteration}.pt"
        import os
        torch.save(save_dict, os.path.join(directory, file_name))

