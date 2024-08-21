import os
import yaml
import argparse
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

parser = argparse.ArgumentParser(description='Process some IMAGEMET.')
parser.add_argument('--data_id', type=str, default="selected2imagenet", help='data id')
parser.add_argument('--min_scls', type=int, default=10, help='Minimum number of subclasses')
parser.add_argument('--split_sizes', type=str, default="50,25", help='Percentiles for splitting')
args = parser.parse_args()

# Read the existing yaml file
data_id = args.data_id
min_num_subclasses = args.min_scls
split_sizes = [int(x) for x in args.split_sizes.split(sep=',')]
info_file = f"mapping/{data_id}_human_checked_remove_replicas_dict.yaml"
out_file = f"mapping/{data_id}_sub_{min_num_subclasses}_split_{split_sizes[0]}-{split_sizes[1]}.yaml"

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

        scores = np.array(info['selected_scores'])
        #divide scores in 3 percentiles
        percentile_1 = np.percentile(scores, split_sizes[0])
        percentile_2 = np.percentile(scores, split_sizes[1])

        # Split the array into three parts
        train = scores >= percentile_1
        test_easy = (scores < percentile_1) & (scores >= percentile_2)
        test_hard = scores < percentile_2

        cls_dict = {
            'scores': info[('selected_scores')],
            'folders': info[('selected_folders')],
            'classes': info[('selected_classes')],
            'train_filter': [bool(label) for label in train],
            'test_easy_filter': [bool(label) for label in test_easy],
            'test_hard_filter': [bool(label) for label in test_hard]
        }
        score_dict[superclass] = cls_dict

# Save the filtered dictionary
with open(out_file, 'w') as file:
    yaml.dump(score_dict, file)
    print(f"Saved filtered info to {out_file}")

#visualize splits for each superclass in a histogram

# Prepare data for plotting
superclasses = list(score_dict.keys())
train_counts = [np.sum(score_dict[sc]['train_filter']) for sc in superclasses]
test_easy_counts = [np.sum(score_dict[sc]['test_easy_filter']) for sc in superclasses]
test_hard_counts = [np.sum(score_dict[sc]['test_hard_filter']) for sc in superclasses]

# Create stacked bar plot
bar_width = 0.7
index = np.arange(len(superclasses))

fig, ax = plt.subplots(figsize=(18, 8))
bar1 = ax.bar(index, train_counts, bar_width, label='Train')
bar2 = ax.bar(index, test_easy_counts, bar_width, bottom=train_counts, label='Test Easy')
bar3 = ax.bar(index, test_hard_counts, bar_width, bottom=np.array(train_counts) + np.array(test_easy_counts), label='Test Hard')

# Add labels and title
ax.set_xlabel('Superclasses')
ax.set_ylabel('Number of Subclasses')
ax.set_title('Number of Subclasses for Each Superclass')
ax.set_xticks(index)
ax.set_xticklabels(superclasses, rotation=45, ha='right')
ax.legend()

# Add text labels
for i in range(len(superclasses)):
    ax.text(i, train_counts[i] / 2, str(train_counts[i]), ha='center', va='center', color='black')
    ax.text(i, train_counts[i] + test_easy_counts[i] / 2, str(test_easy_counts[i]), ha='center', va='center', color='black')
    ax.text(i, train_counts[i] + test_easy_counts[i] + test_hard_counts[i] / 2, str(test_hard_counts[i]), ha='center', va='center', color='black')

# Display the plot
plt.tight_layout()
plt.show()

#save fig
fig.savefig(f"data_stat/{data_id}_sub_{min_num_subclasses}_split_{split_sizes[0]}_{split_sizes[1]}.png")
