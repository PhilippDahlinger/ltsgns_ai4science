import torch
from mp_pytorch.mp import MPFactory


class ProDMPPredictor(torch.nn.Module):
    def __init__(self, num_dof, mp_config: dict):
        super().__init__()

        self.mp_config = mp_config
        self.num_dof = num_dof
        self.dt = self.mp_config["mp_args"]["dt"]
        self._output_size = None

        self.learn_tau = mp_config["learn_tau"]
        if self.learn_tau:
            self.min_tau = mp_config["min_tau"]
            self.max_tau = mp_config["max_tau"]
        # delete min/max tau from config since ProDMP crashes otherwise
        if "min_tau" in mp_config:
            del mp_config["min_tau"]
        if "max_tau" in mp_config:
            del mp_config["max_tau"]

        self.mp_config["num_dof"] = num_dof

        self._trajectory_prediction_times = torch.arange(0, 1, self.dt)
        self.mp_cuda = MPFactory.init_mp(**self.mp_config, device="cuda")
        self.mp_cpu = MPFactory.init_mp(**self.mp_config, device="cpu")

    @property
    def output_size(self):
        if self._output_size is None:
            self._output_size = (
                self.num_dof * self.mp_config["mp_args"]["num_basis"]
                + self.num_dof
                + 1 * self.learn_tau
            )
        return self._output_size

    def get_mp(self, device):
        if device.type == "cuda":
            return self.mp_cuda
        else:
            return self.mp_cpu

    def forward(
        self, pos, vel, basis_weights: torch.Tensor, prediction_times: torch.Tensor | None = None, output_vel: bool = True
    ) -> torch.Tensor:
        if self.learn_tau:
            sigmoid_result = (
                torch.sigmoid(basis_weights[..., 0]) * self.max_tau + self.min_tau
            )
            basis_weights = torch.cat(
                (sigmoid_result.unsqueeze(-1), basis_weights[..., 1:]), dim=-1
            )

        batch_size = pos.shape[:-1]
        device = pos.device

        initial_time = torch.zeros(batch_size, device=device)
        # repeat prediction times for each batch
        if prediction_times is None:
            prediction_times = self._trajectory_prediction_times.repeat(*batch_size, 1).to(device)

        # Predict trajectory
        mp = self.get_mp(device)
        mp.reset()
        mp.update_inputs(
            times=prediction_times,
            init_pos=pos,
            init_vel=vel / self.dt,
            init_time=initial_time,
            params=basis_weights,
        )
        trajectories = mp.get_trajs()

        if output_vel:
            output_pos = trajectories["pos"]
            output_vel = trajectories["vel"] * self.dt
            output = torch.cat((output_pos, output_vel), dim=-1)
        else:
            output = trajectories["pos"]

        # output = output.permute(1, 0, 2).to(device)

        return output
