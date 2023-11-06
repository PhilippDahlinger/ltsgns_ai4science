from typing import Tuple, Union, Optional, Dict

import torch
from gmm_util.gmm import GMM
from hmpn.abstract.abstract_message_passing_base import AbstractMessagePassingBase
from multi_daft_vi.util_multi_daft import create_initial_gmm_parameters
from torch import nn
from torch_geometric.data import Batch

from lts_gns.algorithms.simulators.abstract_graph_network_simulator import AbstractGraphNetworkSimulator
from lts_gns.algorithms.simulators.simulator_util import unpack_node_features
from lts_gns.architectures.decoder.decoder import Decoder
from lts_gns.architectures.decoder.decoder_factory import DecoderFactory
from lts_gns.architectures.prodmp.prodmp import ProDMPPredictor
from lts_gns.architectures.util.chamfer_distance import padded_chamfer_distance

from lts_gns.envs.abstract_gns_environment import AbstractGNSEnvironment
from lts_gns.envs.prodmp_environment import ProDMPEnvironment
from lts_gns.util import keys
from lts_gns.util.config_dict import ConfigDict
from lts_gns.util.own_types import ValueDict
from lts_gns.util.util import node_type_mask


class LTSGNSProDMPSimulator(AbstractGraphNetworkSimulator):
    """
    Simulator with the Message Passing Network (MPN) architecture.
    """

    def _loss(self, predictions: torch.Tensor, labels: torch.Tensor, loss_function_name: str) -> torch.Tensor:
        pass

    def predict_denormalize_and_integrate(self, batch: Batch, **kwargs) -> torch.Tensor:
        pass

    def __init__(self, simulator_config: ConfigDict, example_input_batch: Batch, d_z: int,
                 env: ProDMPEnvironment, device: str):
        """
        Builds the simulator. It predicts the next state of the environment, given the current state and a latent state.
        :param simulator_config:
        :param example_input_batch: To get the dimension of the input.
        """
        self._d_z = d_z
        self._mode = "gnn_step"  # "training mode". May be either "posterior_step" or "gnn_step"
        self._trajectory_length = len(env.trajectories[keys.TRAIN][0])
        self._mp_predictor = self.build_prodmp(example_input_batch=example_input_batch,
                                               simulator_config=simulator_config)
        # decoder for predicting the initial velocity as boundary condition for the ProDMP
        self._vel_decoder = None
        # todo putting this so far up is a bit ugly, but we need the self._d_z to build the decoder in super().__init__
        super().__init__(simulator_config,
                         example_input_batch=example_input_batch,
                         env=env,
                         device=device)
        self._cached_processor_output: torch.Tensor | None = None

        prior_w, prior_mean, prior_cov = create_initial_gmm_parameters(
            n_tasks=1,
            d_z=self._d_z,
            n_components=self.simulator_config.prior.n_components,
            prior_scale=self.simulator_config.prior.prior_scale,
            initial_var=self.simulator_config.prior.initial_var,
        )
        self.prior = GMM(
            log_w=prior_w,
            mean=prior_mean,
            prec=torch.linalg.inv(prior_cov),
            device=device
        )

    def build_prodmp(self, example_input_batch: Batch, simulator_config: ConfigDict) -> ProDMPPredictor:
        mp_config = simulator_config.mp_config.get_raw_dict()
        # assume that all trajs have same length, take the length of the first train traj
        mp_config["mp_args"]["dt"] = 1 / self._trajectory_length
        mp_predictor = ProDMPPredictor(
            num_dof=example_input_batch[keys.POSITIONS].shape[1],
            mp_config=mp_config,
        )
        return mp_predictor

    def _build_decoder(self, example_input_batch: Batch) -> Decoder:
        # overwrite in order to get the correct output dimension (i.e. action_dim)
        decoder = DecoderFactory.build_decoder(decoder_config=self.simulator_config.decoder,
                                               input_dimensions=self._input_dimensions,
                                               action_dim=self._mp_predictor.output_size,
                                               simulator_class=str(type(self)).split(".")[-1].split("'")[0],
                                               device=self._device)
        if self.simulator_config.decoder.velocity_decoder:
            self._vel_decoder = DecoderFactory.build_decoder(decoder_config=self.simulator_config.decoder,
                                                             input_dimensions=self._input_dimensions,
                                                             action_dim=example_input_batch[keys.POSITIONS].shape[-1],
                                                             simulator_class=str(type(self)).split(".")[-1].split("'")[
                                                                 0],
                                                             device=self._device)
        return decoder

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, mode: str):
        self._mode = mode

    @property
    def mp_predictor(self):
        if not hasattr(self, "_mp_predictor"):
            raise ValueError("MP predictor not initialized yet.")
        return self._mp_predictor

    @property
    def decoder(self) -> nn.Module:
        return self._decoder

    @property
    def processor(self) -> AbstractMessagePassingBase:
        return self._processor

    @property
    def _input_dimensions(self) -> Dict[str, int]:
        return {keys.PROCESSOR_DIMENSION: self.simulator_config.gnn.latent_dimension,
                keys.Z_DIMENSION: self._d_z}

    def loss(self, batch: Batch, **kwargs) -> torch.Tensor:
        """
        Computes the loss of the simulator. This is in this case the negative log likelihood part of the ELBO.
        """
        # compute the output of the GNN -> Decoder -> ProDMP
        predictions = self._predict(batch=batch, predict_context_timesteps=True, **kwargs)
        # log likelihood per time step
        predictions = self._combine_tasks_and_time_steps(predictions)
        # prediction has shape (num_samples, selected_context_timesteps, n_vertices, d_world)
        labels = batch[keys.CONTEXT_NODE_POSITIONS]
        # labels has shape (selected_context_timesteps, n_vertices, d_world)
        log_likelihood_per_time_step = self._log_likelihood_per_time_step(predictions=predictions, labels=labels,
                                                                          likelihood_std=1.0)
        # mean over all time steps and z-samples to obtain the loss
        loss = -log_likelihood_per_time_step.mean()
        return loss

    def condition_on_data(self, batch: Batch):
        """
        Specifies on which data the simulator should be conditioned on. Also computes the output of the GNN without backprop.
        The output of the GNN is stored in self._gnn_output and used for the log density computation wrt z.
        :param batch: Data batch generated from a graph env.
        """
        with torch.no_grad():
            # we don't want dropout to predict the posterior. The dropout is just to make the output less dependent
            # on the x, not on the z. But the z has to be correct, hence no dropout.
            processed_batch = self.processor(batch)

            self._cached_processor_output = unpack_node_features(processed_batch, node_type=keys.MESH)
        self._batch = batch

    def evaluate_state(self, predictions: torch.Tensor,
                       reference_labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Computes various metrics for the predicted positions given the internal labels.
        Args:
            predicted_mesh_state: Arbitrary outputs of the simulator. Can e.g., be interpreted as node velocities
            depending on the task.
            reference_labels: If given, the metrics are computed with respect to these labels. Otherwise, the internal

        Returns: Dictionary with the metrics
            "mse": MSE over all samples and time steps.
                Shape: () (float)
            # "best_mse": MSE over all samples and time steps, but only the best latent per time step is considered.
            #     Shape: () (float)
            "log_likelihood": Log likelihood of the predicted positions given the labels.
                Shape: () (float)

        """
        return {keys.MSE: self.mse(predictions=predictions,
                                   reference_pos=reference_labels),
                # keys.BEST_MSE: self.best_mse(predictions=predictions,
                #                              reference_labels=reference_labels),
                keys.MLL: self.log_marginal_likelihood(predictions=predictions,
                                                       reference_labels=reference_labels),
                }

    def mse(self, predictions: torch.Tensor, reference_pos: torch.Tensor) -> torch.Tensor:
        """
        Computes the mean squared error of the predicted positions given the internal labels.
        Implementation of Equation (20) from Volpp. et. al. 2023
        Args:
            predictions: Tensor with shape (num_samples, n_vertices, d_world)
            reference_pos: If given, the metrics are computed with respect to these positions. Otherwise, the internal
              next_mesh_pos positions are used.

        Returns: MSE over all samples and time steps. Shape: () (float)

        """
        # mean prediction over all z samples
        predictions = torch.mean(predictions, dim=0)
        # predicted_positions has shape (n_vertices, d_world)
        # mse error to labels
        # in Eq. (20) they average over time steps, here we average over vertices,
        # which is the same as taking the mean over vertices per
        # time step and then the mean over time steps.
        mse = torch.mean((predictions - reference_pos) ** 2)
        return mse

    def best_mse(self, predictions: torch.Tensor, reference_labels: torch.Tensor) -> torch.Tensor:
        """
        Computes the mean squared error of the predicted positions given the internal labels for all latents z
        and then takes the best one.
        Args:
            predictions: Tensor with shape (num_samples/z, n_vertices, d_world)
            reference_labels: If given, the metrics are computed with respect to these labels. Otherwise, the internal
              labels are used.

        Returns: MSE of the best latent z

        """
        # mean prediction over all z samples
        mse_per_z = torch.mean((predictions - reference_labels) ** 2, dim=(-2, -1))
        best_mse = torch.min(mse_per_z)
        return best_mse

    def log_marginal_likelihood(self, predictions: torch.Tensor,
                                reference_labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Computes the log marginal likelihood of the data given the model parameters.
        Implementation of Equation (19) from Volpp. et. al. 2023
        Args:
            predictions: Tensor with shape (num_samples, n_vertices, d_world)
            reference_labels: If given, the metrics are computed with respect to these labels. Otherwise, the internal
                labels are used.

        Returns:

        """
        log_likelihood_per_time_step = self._log_likelihood_per_time_step(predictions=predictions,
                                                                          labels=reference_labels
                                                                          )
        # sum over all time steps
        log_likelihood = torch.sum(log_likelihood_per_time_step, dim=1)
        # log_likelihood has shape (num_samples,)
        # compute the log marginal likelihood via logsumexp
        log_marginal_likelihood = torch.logsumexp(log_likelihood, dim=0) - torch.log(
            torch.tensor(log_likelihood.shape[0]))
        return log_marginal_likelihood

    def _log_likelihood_per_time_step(self, predictions: torch.Tensor,
                                      labels: torch.Tensor,
                                      likelihood_std: float | None = None,
                                      loss_function_name: str | None = None) -> torch.Tensor:
        """
        Computes the log likelihood of the predicted positions given the labels.
        The labels are the next positions the next time step of the nodes.
        :param predictions: tensor with shape (num_samples, batch_dim/selected_timesteps, n_vertices, d_world).
            Can e.g., be the predicted positions of the simulator, but could also be the model velocities for training.
        :param labels: tensor with shape (batch_dim/selected_timesteps, n_vertices, d_world).
        :param likelihood_std: standard deviation of the gaussian likelihood. If None, the std from the config is used.
        :return: log-likelihood of each time step. Shape: (num_samples, batch_dim/selected_timesteps)
        """
        if loss_function_name is None:
            loss_function_name = self._loss_function_name

        if loss_function_name == "mse":
            # log likelihood of a gaussian with mean extend_label and std given from config
            if likelihood_std is None:
                likelihood_std = self._simulator_config.likelihood.std
            log_likelihood_per_node = - 0.5 * torch.sum((predictions - labels) ** 2 / likelihood_std ** 2, dim=-1)
            # likelihood has shape (num_samples, selected_context_timesteps, n_vertices)
            if self._simulator_config.likelihood.graph_aggregation == "sum":
                aggregator = torch.sum
            elif self._simulator_config.likelihood.graph_aggregation == "mean":
                aggregator = torch.mean
            else:
                raise ValueError(
                    f"Unknown graph aggregation method: {self._simulator_config.likelihood.graph_aggregation}")

            log_likelihood_per_time_step = aggregator(log_likelihood_per_node, dim=-1)
            # now has shape (num_samples, num_time_steps/num_graphs)

        elif loss_function_name == "chamfer":
            if likelihood_std is None:
                likelihood_std = self._simulator_config.likelihood.chamfer_std
            chamfer_loss = padded_chamfer_distance(predictions, labels,
                                                   density_aware=self._simulator_config.chamfer.density_aware,
                                                   forward_only=self._simulator_config.chamfer.forward_only,
                                                   point_reduction=self._simulator_config.likelihood.graph_aggregation)
            log_likelihood_per_time_step = - 0.5 * chamfer_loss / (likelihood_std ** 2)
        else:
            raise ValueError(f"Unknown loss function: {loss_function_name}")

        return log_likelihood_per_time_step

    def _combine_tasks_and_time_steps(self, predictions: torch.Tensor):
        """
        Selects the context time steps per task from the predictions and combines them into one dimension.
        Args:
            predictions: shape (num_samples, num_tasks, num_time_steps, num_nodes_per_task, dim)
            batch: Task batch

        Returns: predictions with shape (num_samples, num_tasks * num_time_steps, num_nodes_per_task, dim)
        """
        # combine num_tasks and num_time_steps into one dimension
        predictions = predictions.reshape(predictions.shape[0], -1, *predictions.shape[3:])
        return predictions

    def _predict(self, z: torch.Tensor | None, batch: Optional[Batch] = None, predict_context_timesteps: bool = False,
                 use_cached_processor_output: bool = True) -> torch.Tensor:
        """
        Predicts the positions of all time steps. Requires a latent z.
        Will use the internal batch if no batch is given. Since the processor calculation of this batch is cached, uses
        the cached processor output if possible.
        Args:
            z: Tensor with shape (num_samples, num_tasks, d_z)
            batch: If given, the internal batch is replaced by this one.
            use_cached_processor_output: If True and no batch is provided, uses the cached processor output.
              Otherwise, calculates the processor output from self._batch if no batch is provided,
              or from batch otherwise

        Returns: Simulator predictions of shape (num_samples, tasks, timesteps, num_nodes_per_task, dim)

        """
        if batch is None:
            batch = self._batch
            if use_cached_processor_output:
                processor_output = self.cached_processor_output
            else:
                processed_batch = self.processor(self._batch)
                processor_output = unpack_node_features(processed_batch, node_type=keys.MESH)
        else:
            processed_batch = self.processor(batch)
            processor_output = unpack_node_features(processed_batch, node_type=keys.MESH)
        example_graph = batch[0]
        mesh_nodes_per_task = example_graph[keys.CONTEXT_NODE_POSITIONS].shape[
            1]  # shape (num_context_points, num_nodes, dim)
        basis_weights = self.decoder(processor_output=processor_output,
                                     z=z,
                                     mesh_nodes_per_task=mesh_nodes_per_task)

        mesh_init_pos = batch[keys.POSITIONS][node_type_mask(batch, keys.MESH)]
        # repeat over all z
        mesh_init_pos = mesh_init_pos.repeat(z.shape[0], 1, 1)
        if self._vel_decoder is not None:
            mesh_init_vel = self._vel_decoder(processor_output=processor_output,
                                              z=z,
                                              mesh_nodes_per_task=mesh_nodes_per_task)
        else:
            mesh_init_vel = torch.zeros_like(mesh_init_pos, device=mesh_init_pos.device)
        if predict_context_timesteps:
            context_timesteps = batch[keys.CONTEXT_INDICES]
            context_timesteps = context_timesteps.reshape(len(batch), example_graph[keys.CONTEXT_SIZES][0])
            context_timesteps = torch.repeat_interleave(context_timesteps, mesh_nodes_per_task, dim=0)
            context_timesteps = context_timesteps[None, :, :].repeat(z.shape[0], 1, 1)
            # normalize into 0-1 range as this is what ProDMP expects
            context_timesteps = context_timesteps / self._trajectory_length
        else:
            context_timesteps = None
        simulator_predictions = self.mp_predictor(mesh_init_pos, mesh_init_vel, basis_weights=basis_weights,
                                                  prediction_times=context_timesteps,
                                                  output_vel=False )
        # shape (num_samples, tasks*num_nodes_per_task, timesteps, dim)
        # entangle tasks and nodes
        simulator_predictions = simulator_predictions.reshape(simulator_predictions.shape[0], len(batch),
                                                              mesh_nodes_per_task, *simulator_predictions.shape[2:])
        # swap last two dimensions
        simulator_predictions = simulator_predictions.permute(0, 1, 3, 2, 4)
        # shape (num_samples, tasks, timesteps, num_nodes_per_task, dim)
        return simulator_predictions

    def log_density(self, z: torch.Tensor | None, compute_grad: bool = False) -> Tuple[
        torch.Tensor, Union[torch.Tensor, None]]:
        """
        Computes the unnormalized task posterior log p(z| data) := log p(data | z)  + log p(z).
        Implementation of Equation (18) from Volpp. et. al. 2023
        :param z: latent variables. Shape: (num_samples, num_tasks, d_z)
        :param compute_grad: If True, the gradient of the log density with respect to z is computed and returned as extra output.
        :return: log density. Shape: (num_samples, num_tasks) or (num_samples, num_tasks), (num_samples, num_tasks, d_z)

        """
        if compute_grad:
            z.requires_grad_(True)
            if z.grad is not None:
                z.grad.zero_()

        log_likelihood = self._log_likelihood(z)
        log_prior_density = self._log_prior_density(z=z)
        log_density = log_likelihood + log_prior_density

        # log_density has shape (num_samples, num_tasks)
        if compute_grad:
            # sum and backward
            torch.sum(log_density).backward()
            # check that the result does not contain any nans
            assert torch.isnan(z.grad).sum() == 0
            assert torch.isnan(log_density).sum() == 0
            return log_density, z.grad
        else:
            return log_density, None

    def _log_likelihood(self, z: torch.Tensor):
        predictions = self._predict(z=z, predict_context_timesteps=True, use_cached_processor_output=True)
        predictions = self._combine_tasks_and_time_steps(predictions)
        if self._loss_function_name == "mse":
            # prediction has shape (num_samples, selected_context_timesteps, n_vertices, d_world)
            labels = self._batch[keys.CONTEXT_NODE_POSITIONS]
            # labels has shape (selected_context_timesteps, n_vertices, d_world)
            log_likelihood_per_time_step = self._log_likelihood_per_time_step(predictions=predictions,
                                                                          labels=labels)
        elif self._loss_function_name == "chamfer":
            labels = self._batch[keys.POINT_CLOUD_POSITIONS]
            log_likelihood_per_time_step = self._log_likelihood_per_time_step(predictions=predictions,
                                                                              labels=labels,
                                                                              )
        else:
            raise ValueError(f"Unknown loss function: {self._loss_function_name}")
        context_sizes = tuple(self._batch[keys.CONTEXT_SIZES])
        if self._simulator_config.likelihood.timestep_aggregation == "sum":
            log_likelihood = torch.stack(
                [torch.sum(t, dim=1) for t in torch.split(log_likelihood_per_time_step, context_sizes, dim=1)],
                dim=1)
        elif self._simulator_config.likelihood.timestep_aggregation == "mean":
            log_likelihood = torch.stack(
                [torch.mean(t, dim=1) for t in torch.split(log_likelihood_per_time_step, context_sizes, dim=1)],
                dim=1)
        else:
            raise ValueError(
                f"Unknown time step aggregation method: {self._simulator_config.likelihood.timestep_aggregation}")
        return log_likelihood

    def _log_prior_density(self, z: torch.Tensor):
        log_prior_density, _ = self.prior.log_density(z=z, compute_grad=False)
        return log_prior_density

    # todo what are all of these doing?
    def get_num_dimensions(self):
        return self._d_z

    def can_sample(self):
        return False

    def sample(self, n: int):
        raise NotImplementedError

    @property
    def cached_processor_output(self) -> torch.Tensor:
        """
        Cached output of the GNN used to train the posterior model. Does not have gradients wrt the GNN parameters.
        Returns:

        """
        if self._cached_processor_output is None:
            raise ValueError("GNN Output is not precomputed. Call 'condition_on_data()' first.")
        return self._cached_processor_output

    def set_train_mode(self, train: bool):
        super().set_train_mode(train)

    def _get_all_state_dicts(self) -> ValueDict:
        gnn_params = self.processor.state_dict()
        decoder_params = self.decoder.state_dict()
        mp_params = self.mp_predictor.state_dict()
        save_dict = {"gnn_params": gnn_params,
                     "decoder_params": decoder_params,
                     "mp_params": mp_params}
        if self._vel_decoder is not None:
            vel_decoder_params = self._vel_decoder.state_dict()
            save_dict["vel_decoder_params"] = vel_decoder_params
        return save_dict

    def _insert_state_dict_params(self, state_dict):
        super()._insert_state_dict_params(state_dict)
        # also load the parameters of the MP predictor and the vel decoder
        mp_params = state_dict.get('mp_params')
        assert mp_params is not None, "MP predictor params not found in the state_dict"
        self.mp_predictor.load_state_dict(mp_params)

        if self._vel_decoder is not None:
            vel_decoder_params = state_dict.get('vel_decoder_params')
            assert vel_decoder_params is not None, "Vel decoder params not found in the state_dict"
            self._vel_decoder.load_state_dict(vel_decoder_params)
