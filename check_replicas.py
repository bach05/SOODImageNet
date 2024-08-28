import os
import yaml
import argparse
from tqdm import tqdm
from collections import defaultdict
import numpy as np

parser = argparse.ArgumentParser(description='Process some IMAGEMET.')
parser.add_argument('--data_id', type=str, default="selected2imagenet", help='data id')
args = parser.parse_args()

# Read the existing yaml file
data_id = args.data_id
info_file = f"mapping/{data_id}_human_checked_dict.yaml"
out_file = f"mapping/{data_id}_human_checked_remove_replicas_dict.yaml"

if os.path.exists(info_file):
    with open(info_file, 'r') as file:
        checked_dict = yaml.safe_load(file)
        print(f"Loaded checked info from {info_file}")
else:
    print(f"Checked file {info_file} does not exist, exiting...")
    exit()

#process checked_dict
for key, value in checked_dict.items():
    if 'classes' in value:
        valid_classes = np.array(checked_dict[key]['classes'])[checked_dict[key]['vis_checks']]
        valid_classes = list(valid_classes[checked_dict[key]['manually_check_images']])
        valid_classes = [str(cls) for cls in valid_classes]

        checked_dict[key]['human_checked_classes'] = valid_classes
        checked_dict[key]['replica_filter'] = [True] * len(checked_dict[key]['human_checked_classes'])

# Create a dictionary to track classes across keys
class_occurrences = defaultdict(list)

# Iterate through the keys and their classes
for key, value in checked_dict.items():
    if 'human_checked_classes' in value:
        for cls in value['human_checked_classes']:
            class_occurrences[cls].append(str(key))

# Check for duplicates
duplicates = {cls: keys for cls, keys in class_occurrences.items() if len(keys) > 1}

# Print the duplicates and prompt user for action
if duplicates:
    print("Found duplicate classes in different keys:")
    for cls, keys in duplicates.items():
        print(f"\nClass '{cls}' found in:")
        for i, key in enumerate(keys):
            print(f"{i + 1} - {key}, tot cls: {len(checked_dict[key]['human_checked_classes'])}")

        # Prompt user for which key to keep
        while True:
            try:
                choice = int(input(f"\nEnter the number of the key to keep for class '{cls}': "))
                if 1 <= choice <= len(keys):
                    break
                else:
                    print(f"Invalid choice. Please enter a number between 1 and {len(keys)}.")
            except ValueError:
                print("Invalid input. Please enter a number.")

        # Delete the class from other keys
        selected_key = keys[choice - 1]
        for j, key in enumerate(keys):
            if key != selected_key:
                checked_dict[key]['replica_filter'][j] = False


    # Update the selected folders and scores
    for key, value in checked_dict.items():
        select_classes = list(np.array(checked_dict[key]['human_checked_classes'])[checked_dict[key]['replica_filter']])
        checked_dict[key]['selected_classes'] = [str(cls) for cls in select_classes]
        selected_folders = list(np.array(checked_dict[key]['folders'])[checked_dict[key]['vis_checks']][checked_dict[key]['manually_check_images']][checked_dict[key]['replica_filter']])
        checked_dict[key]['selected_folders_path'] = [str(fold) for fold in selected_folders]
        #checked_dict[key]['selected_folders'] = [os.path.basename(str(folder)) for folder in selected_folders]
        checked_dict[key]['selected_folders'] = [
            os.path.basename(os.path.dirname(str(folder))) + os.path.sep + os.path.basename(str(folder))
            for folder in selected_folders
        ]
        selected_scores = list(np.array(checked_dict[key]['scores'])[checked_dict[key]['vis_checks']][checked_dict[key]['manually_check_images']][checked_dict[key]['replica_filter']])
        checked_dict[key]['selected_scores'] = [float(score) for score in selected_scores]

    # Save the updated dictionary back to the YAML file
    with open(out_file, 'w') as file:
        yaml.safe_dump(checked_dict, file)

    print("\nUpdated dictionary saved.")
else:
    print("No duplicate classes found.")