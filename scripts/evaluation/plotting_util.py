import os
import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib
import pickle
import re




def plot(experiments, names, colors, root_path, file_name):
    output_path = "/home/philipp/projects/lts_gns/output/plots/tikz"

    data = {exp: {"ggns": []} for exp in experiments}

    for exp in experiments:
        for rep in os.listdir(os.path.join(root_path, exp)):
            for file in os.listdir(os.path.join(root_path, exp, rep)):
                if file == "scalars.pkl":
                    with open(os.path.join(root_path, exp, rep, file), "rb") as f:
                        raw_data = pickle.load(f)
                        data[exp]["ggns"].append(float(raw_data["large_eval"]["all_eval_tasks"]["ggns_mse"]))

    mean_ggns = [np.mean(data[exp]["ggns"], axis=0) for exp in experiments]
    std_ggns = [np.std(data[exp]["ggns"], axis=0) for exp in experiments]
    print(std_ggns)
    # bar plot
    plt.figure()
    plt.grid()
    plt.yscale("log")
    # plt.ylim(1.0e-4, 1.0e-2)
    # add the names to the x-axis

    plt.bar(names, mean_ggns, yerr=std_ggns, color=colors)
    # plt.xticks(rotation=45, ha="right")
    plt.ylabel("Rollout Mean Squared Error")

    tikzplotlib.save(os.path.join(output_path, file_name))
    plt.show()


def plot_last_step(experiments, names, colors, root_path, file_name):
    output_path = "/home/philipp/projects/lts_gns/output/plots/tikz"

    data = {exp: {"last_step_eval": []} for exp in experiments}

    for exp in experiments:
        for rep in os.listdir(os.path.join(root_path, exp)):
            for file in os.listdir(os.path.join(root_path, exp, rep)):
                if file == "scalars.pkl":
                    with open(os.path.join(root_path, exp, rep, file), "rb") as f:
                        raw_data = pickle.load(f)
                        if "deformable_plate" in exp:
                            if "mgn" in exp and "prodmp" not in exp:
                                data[exp]["last_step_eval"].append(
                                    float(raw_data["large_eval"]["all_eval_tasks"]["49-step recomputed_edges_mse"]))
                            else:
                                data[exp]["last_step_eval"].append(float(raw_data["large_eval"]["all_eval_tasks"]["k_49_mse"]))
                        else:
                            if "mgn" in exp and "prodmp" not in exp:
                                data[exp]["last_step_eval"].append(
                                    float(raw_data["large_eval"]["all_eval_tasks"]["99-step recomputed_edges_mse"]))
                            else:
                                data[exp]["last_step_eval"].append(
                                    float(raw_data["large_eval"]["all_eval_tasks"]["k_99_mse"]))

    mean = [np.mean(data[exp]["last_step_eval"], axis=0) for exp in experiments]
    std = [np.std(data[exp]["last_step_eval"], axis=0) for exp in experiments]
    # bar plot
    plt.figure()
    plt.grid()
    plt.yscale("log")
    # plt.ylim(1.0e-4, 1.0e-2)
    # add the names to the x-axis

    plt.bar(names, mean, yerr=std, color=colors)
    # plt.xticks(rotation=45, ha="right")
    plt.ylabel("Last step Mean Squared Error")

    tikzplotlib.save(os.path.join(output_path, file_name))
    plt.show()


def adapt(file_name):
    output_path = "/home/philipp/projects/lts_gns/output/plots/tikz"
    output_file_name = file_name
    # Define the string to search for and the replacement string
    search_string = r',0\) rectangle \(axis'
    replacement_string = r',0.00000001) rectangle (axis'

    # Read the content of the input file
    with open(os.path.join(output_path, file_name), 'r') as input_file:
        file_content = input_file.read()

    # Use regular expression to find and replace the string
    updated_content = re.sub(search_string, replacement_string, file_content)
    updated_content_split = updated_content.split("\n")
    for idx, line in enumerate(updated_content_split):
        if line.startswith("\\begin{axis}"):
            updated_content_split.insert(idx + 1, "width=\\textwidth,")
        if line.startswith("ytick="):
            updated_content_split[idx] = "% " + line
            # insert yminorticks=true
            updated_content_split.insert(idx + 1, "yminorticks=true,")
            updated_content_split.insert(idx + 1, "grid=both,")

            break
    updated_content = "\n".join(updated_content_split)

    # Write the updated content to the output file
    with open(os.path.join(output_path, output_file_name), 'w') as output_file:
        output_file.write(updated_content)

    print("Replacement complete. Output saved to", os.path.join(output_path, output_file_name))



