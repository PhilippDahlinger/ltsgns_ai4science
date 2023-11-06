import copy
import time
from typing import Tuple, Union, Optional, Dict

import torch
from gmm_util.gmm import GMM
from hmpn.abstract.abstract_message_passing_base import AbstractMessagePassingBase
from multi_daft_vi.util_multi_daft import create_initial_gmm_parameters
from torch import nn
from torch_geometric.data import Batch

from lts_gns.algorithms.simulators.abstract_graph_network_simulator import AbstractGraphNetworkSimulator
from lts_gns.algorithms.simulators.simulator_util import unpack_node_features
from lts_gns.architectures.util.chamfer_distance import padded_chamfer_distance
from lts_gns.architectures.util.graph_dropout import GraphDropout
from lts_gns.envs.abstract_gns_environment import AbstractGNSEnvironment
from lts_gns.util import keys
from lts_gns.util.config_dict import ConfigDict
from lts_gns.util.own_types import ValueDict
from lts_gns.util.util import polyak_update


class LTSGNSSimulator(AbstractGraphNetworkSimulator):
    """
    Simulator with the Message Passing Network (MPN) architecture.
    """

    def __init__(self, simulator_config: ConfigDict, example_input_batch: Batch, d_z: int,
                 env: AbstractGNSEnvironment, device: str):
        """
        Builds the simulator. It predicts the next state of the environment, given the current state and a latent state.
        :param simulator_config:
        :param example_input_batch: To get the dimension of the input.
        """
        self._d_z = d_z
        self._use_posterior_target_network = simulator_config.use_posterior_target_network
        self._posterior_target_network_rate = simulator_config.posterior_target_network_rate
        self._mode = "gnn_step"  # "training mode". May be either "posterior_step" or "gnn_step"

        # todo putting this so far up is a bit ugly, but we need the self._d_z to build the decoder in super().__init__
        super().__init__(simulator_config,
                         example_input_batch=example_input_batch,
                         env=env,
                         device=device)
        self._graph_dropout = GraphDropout(simulator_config.graph_dropout)

        if self._use_posterior_target_network:
            # todo do we only want to polyak the decoder or also the GNN?
            self._polyak_decoder = copy.deepcopy(self._decoder)
            self._polyak_processor = copy.deepcopy(self._processor)
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

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, mode: str):
        self._mode = mode
        if mode == "posterior_step":
            if self._use_posterior_target_network:
                polyak_update(params=self._decoder.parameters(),
                              target_params=self._polyak_decoder.parameters(),
                              tau=self._posterior_target_network_rate)
                polyak_update(params=self._processor.parameters(),
                              target_params=self._polyak_processor.parameters(),
                              tau=self._posterior_target_network_rate)
        elif mode == "gnn_step":
            pass
        else:
            raise ValueError(f"Unknown mode {mode}")

    @property
    def decoder(self) -> nn.Module:
        if self._mode == "posterior_step" and self._use_posterior_target_network:
            return self._polyak_decoder
        else:
            return self._decoder

    @property
    def processor(self) -> AbstractMessagePassingBase:
        if self._mode == "posterior_step" and self._use_posterior_target_network:
            return self._polyak_processor
        else:
            return self._processor

    @property
    def _input_dimensions(self) -> Dict[str, int]:
        return {keys.PROCESSOR_DIMENSION: self.simulator_config.gnn.latent_dimension,
                keys.Z_DIMENSION: self._d_z}

    def _loss(self, predictions: torch.Tensor, labels: torch.Tensor, loss_function_name: str) -> torch.Tensor:
        """
        Computes the loss of the simulator. This is in this case the negative log likelihood part of the ELBO.
        computes output with gradients for the GNN loss.
        """
        # log likelihood of the model velocities (therefore normalized)
        log_likelihood_per_time_step = self._log_likelihood_per_time_step(predictions=predictions,
                                                                          labels=labels,
                                                                          likelihood_std=1.0,
                                                                          loss_function_name=loss_function_name
                                                                          )

        # the Gradient of the ELBO is the same as the Gradient of the log likelihood
        # mean over all time steps and z-samples to obtain the loss
        loss = -log_likelihood_per_time_step.mean()
        return loss

    def condition_on_data(self, batch: Batch, task_belonging: ValueDict) -> bool:
        """
        Specifies on which data the simulator should be conditioned on. Also computes the output of the GNN without backprop.
        The output of the GNN is stored in self._gnn_output and used for the log density computation wrt z.
        :param batch: Data batch generated from a graph env.
        :param task_belonging: task belonging generated from a graph env.
        :return: True if everything worked fine.
        """
        with torch.no_grad():
            # we don't want dropout to predict the posterior. The dropout is just to make the output less dependent
            # on the x, not on the z. But the z has to be correct, hence no dropout.
            # batch = self._graph_dropout(batch)
            processed_batch = self.processor(batch)

            self._cached_processor_output = unpack_node_features(processed_batch, node_type=keys.MESH)
        return super().condition_on_data(batch, task_belonging)

    def evaluate_state(self, predicted_mesh_state: Dict[str, torch.Tensor],
                       reference_labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
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
        # todo does the log likelihood make sense here?
        # todo does the mse make sense as a mean over the results for all latents?
        # todo the marginal likelihood is simply the negative of the mse for 1 sample. Does this make sense?
        predictions = predicted_mesh_state[keys.PREDICTIONS]
        if reference_labels is None:
            reference_labels = self.next_mesh_pos
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
        The labels are the next positions of the nodes.
        :param predictions: tensor with shape (num_samples, n_vertices, d_world).
            Can e.g., be the predicted positions of the simulator, but could also be the model velocities for training.
        :param labels: If given, the metrics are computed with respect to these labels.
            Otherwise, the internal next_mesh_pos positions are used.
        :param likelihood_std: standard deviation of the gaussian likelihood. If None, the std from the config is used.
        :return: log-likelihood of each time step. Shape: (num_samples, num_time_steps)
        """
        # todo make this more general

        if labels.ndim == 2:
            # broadcast labels to all z samples. Otherwise assumes that labels are already given for all z samples
            # or that they can be broadcasted to all z samples.
            labels = labels[None, ...]

        # log likelihood of a gaussian with mean extend_label and std given from config
        if likelihood_std is None:
            likelihood_std = self._simulator_config.likelihood.std
        # constant_term = - n_actions / 2 * np.log(2 * np.pi) - n_actions * np.log(likelihood_std)

        if loss_function_name is None:
            loss_function_name = self._loss_function_name

        if loss_function_name == "mse":
            log_likelihood_per_node = - 0.5 * torch.sum((predictions - labels) ** 2 / likelihood_std ** 2, dim=2)
            # likelihood has shape (num_samples, n_vertices/n_nodes)
            if self._simulator_config.likelihood.graph_aggregation == "sum":
                aggregator = torch.sum
            elif self._simulator_config.likelihood.graph_aggregation == "mean":
                aggregator = torch.mean
            else:
                raise ValueError(
                    f"Unknown graph aggregation method: {self._simulator_config.likelihood.graph_aggregation}")

            mesh_nodes_per_time_step = self.task_belonging[keys.MESH_NODES_PER_TIME]
            log_likelihood_per_time_step = torch.stack([aggregator(t, dim=1)
                                                        for t in torch.split(log_likelihood_per_node,
                                                                             mesh_nodes_per_time_step, dim=1)], dim=1)
            # now has shape (num_samples, num_time_steps/num_graphs)

        elif loss_function_name == "chamfer":
            # We use multiple z over batched graphs and only one label. Need to broadcast the label to all z samples
            z_shape = predictions.shape[0]
            predictions = predictions.reshape(predictions.shape[0], -1,
                                              self.task_belonging[keys.MESH_NODES_PER_TIME][0],
                                              predictions.shape[-1])
            labels = labels.reshape(-1, self.task_belonging[keys.MESH_NODES_PER_TIME][0], labels.shape[-1])
            labels = labels.repeat(predictions.shape[0], 1, 1, 1)
            labels = labels.reshape(-1, *labels.shape[2:])

            predictions = predictions.reshape(-1, *predictions.shape[2:])
            chamfer_loss = padded_chamfer_distance(predictions, labels,
                                                   density_aware=self._simulator_config.chamfer.density_aware,
                                                   forward_only=self._simulator_config.chamfer.forward_only,
                                                   point_reduction=self._simulator_config.likelihood.graph_aggregation)
            # todo equality check?
            chamfer_loss = - 0.5 * chamfer_loss / (likelihood_std ** 2)
            log_likelihood_per_time_step = chamfer_loss.reshape(z_shape, -1)
            # predictions have shape (num_samples, num_graphs, n_vertices, d_world)

        else:
            raise ValueError(f"Unknown loss function: {loss_function_name}")

        return log_likelihood_per_time_step

    def predict_denormalize_and_integrate(self, z: torch.Tensor | None, batch: Optional[Batch] = None,
                                          use_cached_processor_output: bool = True) -> Dict[str, torch.Tensor]:
        """
        Predicts the quantities of the next time step, denormalizes the model velocities back to world space
        and integrates them with the current mesh
        Args:
            z:
            batch:
            use_cached_processor_output: If True and no batch is provided, uses the cached processor output.
              Otherwise, calculates the processor output from self._batch if no batch is provided,
              or from batch otherwise


        Returns: A dictionary with the integrated quantities. The keys are the quantities and the values are tensors
            with shape (num_samples, n_vertices, d_world). Must always have a key PREDICTION to calculate losses/metrics

        """
        simulator_predictions = self._predict(z=z, batch=batch,
                                              use_cached_processor_output=use_cached_processor_output)

        # denormalize the velocities
        simulator_predictions = self._model_to_world_normalizer(simulator_predictions)
        if batch is None:
            batch = self._batch
        mesh_state = self._integrate_predictions_fn(quantities_to_integrate=simulator_predictions,
                                                    batch=batch)
        return mesh_state

    def _predict(self, z: torch.Tensor | None, batch: Optional[Batch] = None,
                 use_cached_processor_output: bool = True) -> torch.Tensor:
        """
        Predicts the positions of the next time step. Requires a latent z.
        Will use the internal batch if no batch is given. Since the processor calculation of this batch is cached, uses
        the cached processor output if possible.
        Args:
            z: Tensor with shape (num_samples, num_tasks, d_z)
            batch: If given, the internal batch is replaced by this one.
            use_cached_processor_output: If True and no batch is provided, uses the cached processor output.
              Otherwise, calculates the processor output from self._batch if no batch is provided,
              or from batch otherwise

        Returns: Simulator predictions of shape (num_samples/||z||, num_nodes, action_dimension)

        """
        if batch is None:
            if use_cached_processor_output:
                processor_output = self.cached_processor_output
            else:
                self._batch = self._graph_dropout(self._batch)
                processed_batch = self.processor(self._batch)
                processor_output = unpack_node_features(processed_batch, node_type=keys.MESH)
        else:
            batch = self._graph_dropout(batch)
            processed_batch = self.processor(batch)
            processor_output = unpack_node_features(processed_batch, node_type=keys.MESH)
        mesh_nodes_per_task = self.task_belonging[keys.MESH_NODES_PER_TASK]
        simulator_predictions = self.decoder(processor_output=processor_output,
                                             z=z,
                                             mesh_nodes_per_task=mesh_nodes_per_task)
        # sim_output has shape (num_samples, n_vertices, d_world)
        # compare with labels, problem: the labels are the next positions, the sim_output are the velocities.
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
            return log_density, z.grad
        else:
            return log_density, None

    def _log_likelihood(self, z: torch.Tensor):
        if self._loss_function_name == "mse":
            labels = self.batch_y
            predictions = self._predict(z=z, use_cached_processor_output=True)
        elif self._loss_function_name == "chamfer":
            labels = self.next_mesh_pos
            predictions = self.predict_denormalize_and_integrate(z=z, use_cached_processor_output=True) \
                .get(keys.PREDICTIONS)
        else:
            raise ValueError(f"Unknown loss function: {self._loss_function_name}")

        log_likelihood_per_time_step = self._log_likelihood_per_time_step(predictions=predictions,
                                                                          labels=labels)
        # Take the sum/mean over all time steps of each task
        time_steps_per_task = self.task_belonging[keys.TIME_STEPS_PER_TASK]
        if self._simulator_config.likelihood.timestep_aggregation == "sum":
            log_likelihood = torch.stack(
                [torch.sum(t, dim=1) for t in torch.split(log_likelihood_per_time_step, time_steps_per_task, dim=1)],
                dim=1)
        elif self._simulator_config.likelihood.timestep_aggregation == "mean":
            log_likelihood = torch.stack(
                [torch.mean(t, dim=1) for t in torch.split(log_likelihood_per_time_step, time_steps_per_task, dim=1)],
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
        self._graph_dropout.train(train)
