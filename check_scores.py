import os
import yaml
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Process some IMAGEMET.')
parser.add_argument('--data_id', type=str, default="selected2imagenet", help='data id')
parser.add_argument('--min_scls', type=int, default=10, help='Minimum number of subclasses')
args = parser.parse_args()

# Read the existing yaml file
data_id = args.data_id
min_num_subclasses = args.min_scls
info_file = f"mapping/{data_id}_human_checked_remove_replicas_dict.yaml"
out_file = f"mapping/{data_id}_sub_{min_num_subclasses}.yaml"

if os.path.exists(info_file):
    with open(info_file, 'r') as file:
        checked_dict = yaml.safe_load(file)
        print(f"Loaded checked info from {info_file}")
else:
    print(f"Checked file {info_file} does not exist, exiting...")
    exit()

score_dict = {}
for superclass, info in tqdm(checked_dict.items()):
    if len(info['selected_classes']) >= min_num_subclasses:

        cls_dict = {
            'scores': info[('selected_scores')],
            'folders': info[('selected_folders')],
            'classes': info[('selected_classes')],
        }
        score_dict[superclass] = cls_dict

# Save the filtered dictionary
with open(out_file, 'w') as file:
    yaml.dump(score_dict, file)
    print(f"Saved filtered info to {out_file}")

