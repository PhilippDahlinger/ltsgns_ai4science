import numpy as np
import pandas as pd

import wandb
import plotly.express as px

wandb.init(project="test_wandb_tables")

wandb_log_dict = {}

# TODO: Real logging values
logging_dict = {"x": np.arange(10), "y": np.arange(10) ** 2}
data = pd.DataFrame.from_dict(logging_dict)
fig = px.line(data, x="x", y="y", title="ELBO over time")
wandb_log_dict["Task0/elbo_plot"] = fig
wandb.log(wandb_log_dict)

logging_dict = {"x": np.arange(10, 20), "y": -np.arange(10, 20) ** 2}
data = pd.DataFrame.from_dict(logging_dict)
fig = px.line(data, x="x", y="y", title="ELBO over time")
wandb_log_dict["Task0/elbo_plot"] = fig
wandb.log(wandb_log_dict)