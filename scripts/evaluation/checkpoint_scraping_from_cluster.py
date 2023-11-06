import argparse
import os
import shutil

# argparse
parser = argparse.ArgumentParser()
parser.add_argument('exp_folder', help="exp folder in /output/cw2_data to search for checkpoints in")
parser.add_argument("-o", "--output_folder", help="output folder to copy checkpoints to", default="/home/philipp/projects/lts_gns/output/scraped_checkpoints")
parser.add_argument('-s', '--string-list', nargs='+', default=[], help="List of strings to search for in Root folders")

args = parser.parse_args()

string_list = args.string_list
exp_folder = args.exp_folder
output_folder = args.output_folder
# root path
root = "/home/philipp/pool_look/code/lts_gns/output/cw2_data"

path = os.path.join(root, exp_folder)
print("----------------------------")
print("Welcome to the Scraper10000, your personal assistance of scraping checkpoints from the cluster")
print("\n")
print("Searching for checkpoints in: ", path)
print("Searching for strings: ", string_list)
print("----------------------------")


if not os.path.exists(path):
    raise FileNotFoundError("Make sure to start this script from the root folder 'DAVIS' of the project")

valid_exps = []

# search for checkpoints
for dir in os.listdir(path):
    if not os.path.isdir(os.path.join(path, dir)):
        continue
    if len(string_list) != 0:
        for prefix in string_list:
            if prefix in dir:
                valid_exps.append(dir)
                break
    else:
        valid_exps.append(dir)
valid_exps = sorted(valid_exps)
print(f"Found in total {len(valid_exps)} experiments: ")
for exp in valid_exps:
    print(exp)
print("Starting copying files...")
print("----------------------------")
output_path = os.path.join(root, output_folder, exp_folder)
print(f"Creating output folder (if it does not exist yet):  {output_path}")
os.makedirs(output_path, exist_ok=True)
print("----------------------------")

no_config_exps = []
not_loaded_exps = []


for exp in valid_exps:
    print("Starting copying files for experiment: ", exp)
    exp_path = os.path.join(path, exp, "log")
    exp_output_path = os.path.join(output_path, exp, "log")
    os.makedirs(exp_output_path, exist_ok=True)
    for rep in os.listdir(exp_path):
        try:
            checkpoint_folder = os.path.join(exp_path, rep, "checkpoints")
            checkpoint_files = os.listdir(checkpoint_folder)
        except FileNotFoundError:
            print(f"Could not find checkpoint folder for exp {exp_path} and repetition {rep}. Skipping...")
            break
        output_checkpoint_folder = os.path.join(exp_output_path, rep, "checkpoints")
        os.makedirs(output_checkpoint_folder, exist_ok=True)
        # copy checkpoint files
        for file in checkpoint_files:
            try:
                shutil.copyfile(os.path.join(checkpoint_folder, file), os.path.join(output_checkpoint_folder, file))
            except FileNotFoundError:
                print(f"Could not find checkpoint {file} in {checkpoint_folder}. Skipping")
                not_loaded_exps.append((exp, rep))
        # copy config file
        try:
            shutil.copyfile(os.path.join(exp_path, rep, "config.yaml"), os.path.join(exp_output_path, rep, "config.yaml"))
        except FileNotFoundError:
            print(f"Could not find config file in {exp_path}. Skipping...")
            no_config_exps.append((exp, rep))
        print(f"Copying files for repetition {rep} done")
    print("----------------------------")

# make missing exp list unique
no_config_exps = list(set(no_config_exps))
not_loaded_exps = list(set(not_loaded_exps))
print("SUMMARY:")
print(f"No config file found for {len(no_config_exps)} of all experiments:")
for exp in no_config_exps:
    print(exp)
print(f"No checkpoint file found for {len(not_loaded_exps)} of all experiments:")
for exp in not_loaded_exps:
    print(exp)
print("----------------------------")









