# Training scalars. Includes metrics and losses
ACCURACY = "acc"
TOTAL_LOSS = "loss"
CROSS_ENTROPY = "ce"
BINARY_CROSS_ENTROPY = "bce"
MSE = "mse"
RMSE = "rmse"
MAE = "mae"
MLL = "mll"  # marginal log likelihood
BEST_MSE = "best_mse"

# Names for graph nodes and edges
COLLIDER = "collider"
COLLIDER_VELOCITY = "collider_velocity"
MESH = "mesh"
NEXT_MESH_POS = "next_mesh_pos"
COLLIDER_COLLIDER = "edge_collider_collider"
MESH_MESH = "edge_mesh_mesh"
COLLIDER_MESH = "edge_collider_mesh"
MESH_COLLIDER = "edge_mesh_collider"
WORLD_MESH = "edge_world_mesh"
MESH_FACES = "mesh_faces"
FIXED_MESH = "fixed_mesh"
FIXED_MESH_INDICES = "fixed_mesh_indices"
COLLIDER_FACES = "collider_faces"
POISSON_RATIO = "poisson_ratio"
FIXED_NODE_IDS = "fixed_node_ids"
FIXED_NODE_POSITIONS = "fixed_node_positions"
POINT_CLOUD_POSITIONS = "point_cloud_positions"
POINT_CLOUD_COLORS = "point_cloud_colors"
# ProDMP stuff
CONTEXT_SIZES = "context_sizes"
CONTEXT_INDICES = "context_indices"
CONTEXT_NODE_POSITIONS = "context_node_positions"
TRAJECTORY_INDICES = "trajectory_indices"
ANCHOR_INDICES = "anchor_indices"

## information for things that are not part of the graph, but can be used for visualization
VISUAL_COLLDER = "visual_collider"
VISUAL_COLLIDER_FACES = "visual_collider_faces"

LABEL = "label"
INITIAL_MESH_POSITIONS = "initial_mesh_positions"

# Names to identify the different graphs
MESH_NODES_PER_TASK = "mesh_nodes_per_task"
MESH_NODES_PER_TIME = "mesh_nodes_per_time_step"
TASK_INDICES = "task_indices"
TIME_STEPS_PER_TASK = "time_steps_per_task"
TIME_STEPS_PER_SUBTASK = "time_steps_per_subtask"
MESH_EDGE_INDEX = "mesh_edge_index"
COLLIDER_EDGE_INDEX = "collider_edge_index"

# Mesh States
POSITIONS = "pos"
PREDICTIONS = "pred"
VELOCITIES = "vel"

# in- and output features
PROCESSOR_DIMENSION = "processor_dimension"
Z_DIMENSION = "z_dimension"
R_DIMENSION = "r_dimension"

# Processing
PREPROCESSED_GRAPHS = "preprocessed_graphs"

# Recording
FINAL = "final"
FIGURES = "figure"
VIDEO_ARRAYS = "videos"
SCALARS = "scalars"
ADDITIONAL_PLOTS = "additional_plots"
VISUALIZATIONS = "visualizations"
NETWORK_HISTORY = "network_history"
SMALL_EVAL = "small_eval"
LARGE_EVAL = "large_eval"
ALL_EVAL_TASKS = "all_eval_tasks"
ALL_TRAIN_TASKS = "all_train_tasks"
POSTERIOR_LOGGING = "posterior_logging"

# Train/Test/Val
TRAIN = "train"
TEST = "test"
VAL = "val"

# HMPN
SRC = "src"
DEST = "dest"
AGENT = "agent"

# Save and Load
STATEDICT = "mpn_simulator_state_dict"
