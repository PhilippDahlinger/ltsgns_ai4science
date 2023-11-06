import pickle

import numpy as np
import torch
from tqdm import tqdm


def get_dataset(split):
    with open(f"../../../datasets/lts_gns/deformable_plate/deformable_plate_{split}.pkl", "rb") as file:
        rollout_data = pickle.load(file)

    dataset = []

    for trajectory in tqdm(rollout_data):
        for pc, mesh in zip(trajectory["pcd_points"], trajectory["nodes_grid"]):
            # all diffs between pc and mesh
            diffs = pc[:, None] - mesh
            # closest mesh point for each pc point
            closest_mesh_idx = np.argmin(np.linalg.norm(diffs, axis=2), axis=1)
            closest_mesh_point = mesh[closest_mesh_idx]
            displacement = pc - closest_mesh_point
            distance = np.linalg.norm(displacement, axis=1)
            dataset.append({"pc": torch.tensor(pc, dtype=torch.float32),
                            "displacement": torch.tensor(displacement, dtype=torch.float32),
                            "distance": torch.tensor(distance, dtype=torch.float32),
                            "mesh": torch.tensor(mesh, dtype=torch.float32)})

    # save dataset
    with open(f"../../../datasets/lts_gns/deformable_plate/supervised_pc_dataset_{split}.pkl", "wb") as file:
        pickle.dump(dataset, file)
    return dataset




if __name__ == "__main__":
    get_dataset("train")
    get_dataset("val")
    get_dataset("test")
