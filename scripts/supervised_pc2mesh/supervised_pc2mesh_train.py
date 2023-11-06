import os
import pickle

import numpy as np
import torch
import torch_geometric
import wandb
from hmpn.common.hmpn_util import make_batch
from hmpn.get_hmpn import get_hmpn_from_graph
from matplotlib import pyplot as plt
from torch_geometric import transforms
from tqdm import tqdm

from cw2 import experiment, cw_error, cluster_work
from cw2.cw_data import cw_logging

from lts_gns.envs.util.edge_computation import create_radius_edges


class SupervisedGNN(torch.nn.Module):
    def __init__(self, batch, latent_dimension=32, output_dimension=1):
        super().__init__()
        self.gnn = self._get_gnn(batch, latent_dimension)
        self.decoder = self._get_decoder(latent_dimension, output_dimension)

    def forward(self, data):
        data = self.gnn(data)
        data = self.decoder(data.x)
        return data

    def _get_decoder(self, latent_dimension, output_dimension):
        decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dimension, latent_dimension),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(latent_dimension, output_dimension),
        )
        decoder.to("cuda")
        return decoder

    def _get_gnn(self, batch, latent_dimension):
        base_config = {
            "assert_graph_shapes": False,
            "scatter_reduce": "mean",
            # how to reduce the edge features for node updates, and the edge and node features for
            # global updates
            # Can be either a single operation or any list of the following: "mean", "sum", "min" or "max", "std".
            "create_graph_copy": True,  # whether to create a copy of the used graph before the forward pass or not
            "stack": {
                "layer_norm": "inner",  # which kind of layer normalization to use. null/None for no layer norm,
                "num_steps": 10,
                "residual_connections": "inner",
                "mlp": {
                    "activation_function": "leakyrelu",
                    "num_layers": 1,
                    "add_output_layer": False,
                }
            }
        }
        gnn = get_hmpn_from_graph(example_graph=batch, latent_dimension=latent_dimension, base_config=base_config,
                                  unpack_output=False)
        gnn.to("cuda")
        return gnn


def open_dataset():
    with open("../datasets/lts_gns/deformable_plate/supervised_pc_dataset_train.pkl", "rb") as file:
        train_dataset = pickle.load(file)
    with open("../datasets/lts_gns/deformable_plate/supervised_pc_dataset_val.pkl", "rb") as file:
        val_dataset = pickle.load(file)
    return train_dataset, val_dataset


def build_graph_from_pc(pc, y, mesh, include_mesh=False):
    edges = create_radius_edges(radius=0.1, source_nodes=pc, source_shift=0)

    data_transform = transforms.Compose([transforms.Cartesian(norm=False, cat=True),
                                         transforms.Distance(norm=False, cat=True)])
    # build Data object
    if include_mesh:
        data = torch_geometric.data.Data(  # x=torch.empty((pc.shape[0], 0)),
            x=pc,
            edge_index=edges, pos=pc, y=y, mesh=mesh)
    else:
        data = torch_geometric.data.Data(  # x=torch.empty((pc.shape[0], 0)),
            x=pc,
            edge_index=edges, pos=pc, y=y)
    data = data_transform(data)
    return data


def plot_mesh(data):
    import matplotlib.pyplot as plt
    plt.scatter(data.pos[:, 0], data.pos[:, 1])
    # edges
    for edge in data.edge_index.T:
        plt.plot(data.pos[edge, 0], data.pos[edge, 1], c="black")
    plt.show()


class MyExperiment(experiment.AbstractExperiment):
    def initialize(self, config: dict, rep: int, logger: cw_logging.LoggerArray) -> None:
        self.batch_size = 100
        self.num_epochs = 100000
        self.vis_indices = [0, 20, 50, 70, 100]
        self.predict_distance = False
        torch.manual_seed(42 + rep)
        np.random.seed(42 + rep)
        self.train_dataset, self.val_dataset = open_dataset()
        self.model = SupervisedGNN(self.train_dataset, latent_dimension=64,
                                   output_dimension=1 if self.predict_distance else 2)
        self.name = "kluster_displacement_deep_and_thick_2"
        wandb.init(
            project="supervised_pc2mesh",
            name=self.name,
        )

    def run(self, config: dict, rep: int, logger: cw_logging.LoggerArray) -> None:
        # Perform your existing task
        self.train(self.train_dataset, self.val_dataset, self.model, run_name=self.name)

    def train(self, train_dataset, val_dataset, model, run_name="run_01"):
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
        loss_fn = torch.nn.MSELoss()
        train_dataset_graphs = []
        for instance in tqdm(train_dataset, desc="Building train graphs"):
            pc = instance["pc"]
            if self.predict_distance:
                y = instance["distance"].reshape(-1, 1)
            else:
                y = instance["displacement"]
            mesh = instance["mesh"]
            data = build_graph_from_pc(pc, y, mesh, include_mesh=True)
            train_dataset_graphs.append(data)
        val_dataset_graphs = []
        for instance in tqdm(val_dataset, desc="Building val graphs"):
            pc = instance["pc"]
            if self.predict_distance:
                y = instance["distance"].reshape(-1, 1)
            else:
                y = instance["displacement"]
            mesh = instance["mesh"]
            data = build_graph_from_pc(pc, y, mesh, include_mesh=True)
            val_dataset_graphs.append(data)
        train_dataset = train_dataset_graphs
        val_dataset = val_dataset_graphs
        _ = [x.to("cuda") for x in train_dataset]
        _ = [x.to("cuda") for x in val_dataset]
        for epoch in (pbar := tqdm(range(self.num_epochs))):
            logs = {"vis": {}}
            losses = []
            for i in range(0, len(train_dataset), self.batch_size):
                batch = train_dataset[i:i + self.batch_size]
                batch = make_batch(batch)
                output = model(batch)
                loss = loss_fn(output, batch.y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            loss2 = np.mean(losses)
            pbar.set_description(f"Epoch {epoch}: {loss2}")
            logs["train_loss"] = loss2
            if epoch % 15 == 0:
                # save checkpoint of model
                save_path = f"output/checkpoints/{run_name}"
                os.makedirs(save_path, exist_ok=True)
                torch.save(model.state_dict(), save_path + f"/model_{epoch}.pt")
                # evaluate on validation set
                losses = []
                for i in range(0, len(val_dataset), self.batch_size):
                    with torch.no_grad():
                        batch = val_dataset[i:i + self.batch_size]
                        batch = make_batch(batch)
                        output = model(batch)
                        loss = loss_fn(output, batch.y)
                        losses.append(loss.item())
                loss2 = np.mean(losses)
                logs["val_loss"] = loss2

                for idx in self.vis_indices:
                    import plotly.graph_objs as go
                    prediction = model(make_batch([val_dataset[idx]])).detach().cpu().numpy().squeeze()
                    pos = val_dataset[idx].pos.detach().cpu().numpy()
                    mesh = val_dataset[idx].mesh.detach().cpu().numpy()
                    # Create the scatter trace for the main data points
                    if self.predict_distance:
                        scatter_trace = go.Scatter(x=pos[:, 0], y=pos[:, 1], mode="markers",
                                                   marker=dict(color=prediction, colorscale='gray',
                                                               colorbar=dict(title='Prediction')))
                    else:
                        updated_pos = pos - prediction
                        scatter_trace = go.Scatter(x=updated_pos[:, 0], y=updated_pos[:, 1], mode="markers",
                                                   marker=dict(color="blue"))

                    # Create the scatter trace for the mesh points
                    mesh_trace = go.Scatter(x=mesh[:, 0], y=mesh[:, 1], mode="markers",
                                            marker=dict(color="red", symbol="x"))

                    # Add all the traces to the figure
                    fig = go.Figure(data=[scatter_trace, mesh_trace])
                    logs["vis"][f"fig_{idx}"] = fig
                logs["epoch"] = epoch
                wandb.log(logs, step=epoch)

    def finalize(self, surrender: cw_error.ExperimentSurrender = None, crash: bool = False):
        # Skip for Quickguide
        pass


if __name__ == "__main__":
    # Give the MyExperiment Class, not MyExperiment() Object!!
    cw = cluster_work.ClusterWork(MyExperiment)
    # RUN!
    cw.run()
