import pickle

import numpy as np
import wandb2numpy
import matplotlib.pyplot as plt
import pandas as pd

config = {
    "DEFAULT":
        {"entity": "philippdahlinger",
         "project": "LTS_GNS_deformable_plate",
         "output_path": "output/wandb_export", },

    "exp_1":
        {
            "fields": ["all_eval_tasks/10-step recomputed_edges_mse", "all_eval_tasks/5-step recomputed_edges_mse", "all_eval_tasks/1-step recomputed_edges_mse"],
            "groups": ["p2512_lts_gns_deformable_plate"],
            "history_samples": "all",
        },
    "exp_2":
        {
            "fields": ["all_eval_tasks/10-step recomputed_edges_mse", "all_eval_tasks/5-step recomputed_edges_mse",
                       "all_eval_tasks/1-step recomputed_edges_mse"],
            "groups": ["p2518_lts_gns_deformable_plate_original_1_10"],
            "history_samples": "all",
        },

    "exp_3":
        {
            "fields": ["all_eval_tasks/10-step recomputed_edges_mse", "all_eval_tasks/5-step recomputed_edges_mse", "all_eval_tasks/1-step recomputed_edges_mse"],
            "groups": ["p3307_mgn_deformable_plate"],
            "history_samples": "all",
        },
}

# data_dict_1, config_list = wandb2numpy.export_data(config)
# # dump dict to file
# with open("data_dict_1.pkl", "wb") as f:
#     pickle.dump(data_dict_1, f)

# load dict
with open("data_dict_1.pkl", "rb") as f:
    data_dict_1 = pickle.load(f)

title_dict = {
    "all_eval_tasks/10-step recomputed_edges_mse": "10-step MSE",
    "all_eval_tasks/5-step recomputed_edges_mse": "5-step MSE",
    "all_eval_tasks/1-step recomputed_edges_mse": "1-step MSE",
}


exp_1 = data_dict_1["exp_1"]
exp_2 = data_dict_1["exp_2"]
exp_3 = data_dict_1["exp_3"]
for key in data_dict_1["exp_1"]:
    lts_gns_results = []
    lts_less_context = []
    mgn_results = []
    for seed in range(len(exp_1[key])):
        lts_gns_nan = exp_1[key][seed]
        lts_gns_results.append(np.interp(np.arange(len(lts_gns_nan)),
                  np.arange(len(lts_gns_nan))[np.isnan(lts_gns_nan) == False],
                  lts_gns_nan[np.isnan(lts_gns_nan) == False]))
        lts_less_context_nan = exp_2[key][seed]
        lts_less_context.append(np.interp(np.arange(len(lts_less_context_nan)),
                    np.arange(len(lts_less_context_nan))[np.isnan(lts_less_context_nan) == False],
                    lts_less_context_nan[np.isnan(lts_less_context_nan) == False]))

        mgn_nan = exp_3[key][seed]
        mgn_results.append(np.interp(np.arange(len(mgn_nan)),
                    np.arange(len(mgn_nan))[np.isnan(mgn_nan) == False],
                    mgn_nan[np.isnan(mgn_nan) == False]))

    lts_gns_result = np.array(lts_gns_results)[:, :1000]
    lts_less_context_result = np.array(lts_less_context)[:, :1000]
    mgn_result = np.array(mgn_results)[:, :1000]
    mean_lts_gns = lts_gns_result.mean(axis=0)
    mean_mgn = mgn_result.mean(axis=0)
    mean_lts_less_context = lts_less_context_result.mean(axis=0)
    std_lts_gns = lts_gns_result.std(axis=0)
    std_mgn = mgn_result.std(axis=0)
    std_lts_less_context = lts_less_context_result.std(axis=0)
    plt.plot(mean_lts_gns, label="LTS-GNS")
    plt.fill_between(np.arange(len(mean_lts_gns)), mean_lts_gns - std_lts_gns, mean_lts_gns + std_lts_gns, alpha=0.2)
    plt.plot(mean_lts_less_context, label="LTS-GNS less context")
    plt.fill_between(np.arange(len(mean_lts_less_context)), mean_lts_less_context - std_lts_less_context, mean_lts_less_context + std_lts_less_context, alpha=0.2)
    plt.plot(mean_mgn, label="MGN")
    plt.fill_between(np.arange(len(mean_mgn)), mean_mgn - std_mgn, mean_mgn + std_mgn, alpha=0.2)

    plt.title(title_dict[key])
    plt.legend()
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(f"output/plots/{title_dict[key]}.png", dpi=500)
    plt.close()
#
# train_loss = data_dict_1["exp_1"]["scalars.train/loss"][0]
# test_loss = data_dict_1["exp_1"]["scalars.eval/loss"][0]
# ten_steps_loss = data_dict_1["exp_1"]["scalars.eval/10-step eval"][0]
#
# # insert values in nan spots
# train_loss = pd.Series(train_loss).fillna(limit=100, method="ffill").to_numpy()
# test_loss = pd.Series(test_loss).fillna(limit=100, method="bfill").to_numpy()
# ten_steps_loss = pd.Series(ten_steps_loss).fillna(limit=100, method="ffill").to_numpy()
#
# plt.plot(train_loss, label="Train Loss")
# plt.plot(test_loss, label="Eval Loss")
# plt.plot(ten_steps_loss, label="10-step Loss")
#
# plt.legend()
# plt.yscale("log")
# plt.tight_layout()
# plt.savefig("output/plots/jour_fixe_3.png", dpi=1000)
# # results = data_dict_1["exp_1"]
# # results.update(data_dict_1["exp_2"])
#
# print("stop")
