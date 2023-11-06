import io
import os
from typing import Union, Any

import wandb
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from plotly.graph_objs import Figure as PlotlyFigure


def get_job_type(entity: str, project: str, group: str) -> str:
    """
    Starts the WandB API and searches for the group in the given project of the given entity.
    If there are no job types (e.g. no runs) returns a job type "run_00"
    Otherwise, it finds the highest job type number x and returns run_{x+1:02d}.
    """
    api = wandb.Api()
    group_runs = api.runs(f"{entity}/{project}", filters={"group": group})
    job_types = list(set([r.job_type for r in group_runs]))
    if len(job_types) == 0:
        return "run_00"
    else:
        indices = [int(job_type[job_type.rfind("_") + 1:]) for job_type in job_types]
        indices.sort()
        last_idx = indices[-1]
        new_job_type = f"run_{last_idx + 1:02d}"
        return new_job_type


def reset_wandb_env():
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
        "WANDB_START_METHOD",
    }
    for k, v in os.environ.items():
        if k.startswith("WANDB_") and k not in exclude:
            del os.environ[k]


def wandbfy(vis_figure: Union[plt.Figure, FuncAnimation, str, PlotlyFigure, Any]) \
        -> Union[wandb.Image, wandb.Video, wandb.Html, Any]:
    """
    Converts the given figure to a wandb object that is loggable.
    Args:
        vis_figure:

    Returns:

    """
    if isinstance(vis_figure, plt.Figure):
        return wandb.Image(vis_figure)
    elif isinstance(vis_figure, FuncAnimation):
        try:
            return wandb.Video(vis_figure.to_html5_video())
            # todo here, debug and check.
        except Exception as e:
            print(f"Error converting animation to html: {e}. Resorting to (big) JS animation.")
            return wandb.Html(vis_figure.to_jshtml())

    elif isinstance(vis_figure, str):
        if vis_figure.endswith(".gif"):
            return wandb.Video(vis_figure)
        elif vis_figure.endswith(".png"):
            return wandb.Image(vis_figure)
    elif isinstance(vis_figure, PlotlyFigure) and len(vis_figure.frames) > 0:
        # if the plotly figure contains an animation (i.e., has non-empty frames), convert it to a html file
        # such that this animation can be properly displayed in the wandb dashboard
        from plotly.io import to_html
        return wandb.Html(to_html(vis_figure, include_plotlyjs='cdn'))
    else:
        # just log it
        return vis_figure
