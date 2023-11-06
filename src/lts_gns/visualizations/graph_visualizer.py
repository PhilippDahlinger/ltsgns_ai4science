from pathlib import Path
from typing import List

import numpy as np
import plotly
from torch_geometric.data import Data

from lts_gns.util.config_dict import ConfigDict
from lts_gns.visualizations.matplotlib_visualizations import visualize_trajectory as visualize_trajectory_matplotlib
from lts_gns.visualizations.plotly_visualizations import visualize_trajectory as visualize_trajectory_plotly


class GraphVisualizer:

    def __init__(self, visualization_config: ConfigDict):
        self._visualization_config: ConfigDict = visualization_config

        self._backend = visualization_config.backend
        if self._backend == "matplotlib":
            self._vis_fn = visualize_trajectory_matplotlib
        elif self._backend == "plotly":
            self._vis_fn = visualize_trajectory_plotly
        else:
            raise ValueError(f"Unknown backend {self._backend}.")

    def visualize_trajectory(self,
                             trajectory: List[Data],
                             ground_truth_trajectory: List[Data] | None = None,
                             context: bool = False):
        """
        Visualize a trajectory of graphs.
        Args:
            trajectory: List of Data objects forming the trajectory. Assumes that the first graph is the initial graph
              and that the topology of the graphs does not change over time.
            ground_truth_trajectory: List of Data objects forming the ground truth trajectory. If None, no ground truth
                is plotted.
            context:
        Returns: Path to the saved animation if save_animation is True, otherwise None.

        """
        # subsample trajectory if not all frames should be visualized
        num_frames = self._visualization_config.num_frames
        if num_frames > 0 & num_frames < len(trajectory):  # Subsample the trajectory
            frame_indices = np.linspace(0, len(trajectory) - 1, num_frames).astype(int)
            trajectory = [trajectory[i] for i in frame_indices]
            if ground_truth_trajectory is not None:
                ground_truth_trajectory = [ground_truth_trajectory[i] for i in frame_indices]

        fig = self._vis_fn(trajectory=trajectory,
                           visualization_config=self._visualization_config,
                           fps=self._visualization_config.fps,
                           limits=self._visualization_config.limits,
                           ground_truth_trajectory=ground_truth_trajectory,
                           context=context)
        return fig

    def save_animation(self, animation, save_path: str | Path, filename):
        import plotly.graph_objects as go
        import plotly.offline as pyo
        from matplotlib import animation as plt_anim

        if isinstance(save_path, str):
            save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        file_path = save_path / filename

        if isinstance(animation, go.Figure):  # plotly
            file_path = str(file_path)
            if file_path.endswith(".gif"):
                # replace with .html
                file_path = file_path[:-4] + ".html"
            pyo.plot(animation, filename=file_path, auto_open=False)
        elif isinstance(animation, plt_anim.FuncAnimation):  # matplotlib
            animation.save(file_path, writer='pillow',
                           fps=self._visualization_config.fps,
                           dpi=self._visualization_config.matplotlib.dpi
                           )

        else:
            raise ValueError(f"Unknown return type of visualization function: {type(animation)}")

    @property
    def visualization_config(self) -> ConfigDict:
        return self._visualization_config
