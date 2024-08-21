import numpy as np
import yaml
import os
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Process some IMAGEMET.')
parser.add_argument('--data_id', type=str, default="selected2imagenet_sub_10_split_50-25", help='data id')

args = parser.parse_args()

# Read the existing yaml file
data_id = args.data_id
data_file = f"mapping/{data_id}.yaml"
with open(data_file, 'r') as file:
    data_dict = yaml.safe_load(file)

info_file = f"mapping/{data_id}_equalized.yaml"
if os.path.exists(info_file):
    with open(info_file, 'r') as file:
        cmd = input(f"Checked file {info_file} exists, do you want to load it? y/n \n")
        if cmd == 'n':
            print("OVERWRITING the checked file")
            checked_dict = {}
        else:
            checked_dict = yaml.safe_load(file)
            print(f"Loaded checked info from {info_file}")
else:
    print(f"Checked file {info_file} does not exist, creating a new one")
    checked_dict = {}

#check the number of subclasses in each class
min_subclasses_number = np.inf
max_subclasses_number = 0

for superclass, info in data_dict.items():
    subclasses = info['classes']
    num_subclasses = len(subclasses)
    print(f"Superclass: {superclass}, Number of subclasses: {num_subclasses}")
    if num_subclasses < min_subclasses_number:
        min_subclasses_number = num_subclasses
    if num_subclasses > max_subclasses_number:
        max_subclasses_number = num_subclasses

print("************************************")
print(f"Minimum number of subclasses: {min_subclasses_number}")
print(f"Maximum number of subclasses: {max_subclasses_number}")

th_max_subclasses = 4*min_subclasses_number

#equalize the number of subclasses
for superclass, info in data_dict.items():
    subclasses = info['classes']
    num_subclasses = len(subclasses)

    checked_dict[superclass] = info

    #remove subclasses when num_subclasses > th_max_subclasses and update train, test_easy, test_hard filters
    if num_subclasses > th_max_subclasses:
        print(f"Superclass: {superclass}, Number of subclasses: {num_subclasses}")
        print(f"Removing {num_subclasses - th_max_subclasses} subclasses")
        print(f"Before: {info['train_filter'].count(True)} train, {info['test_easy_filter'].count(True)} test_easy, {info['test_hard_filter'].count(True)} test_hard")

        #generate random bool values to keep or discard from original list
        indexes = np.random.choice(num_subclasses, th_max_subclasses, replace=False)
        bool_indexes = [True if i in indexes else False for i in range(num_subclasses)]
        train_filter = [bool_indexes[i] * info['train_filter'][i] for i in range(num_subclasses)]
        test_easy_filter = [bool_indexes[i] * info['test_easy_filter'][i] for i in range(num_subclasses)]
        test_hard_filter = [bool_indexes[i] * info['test_hard_filter'][i] for i in range(num_subclasses)]

        # subclasses = subclasses[:th_max_subclasses]
        # train_filter = info['train_filter'][:th_max_subclasses]
        # test_easy_filter = info['test_easy_filter'][:th_max_subclasses]
        # test_hard_filter = info['test_hard_filter'][:th_max_subclasses]

        print(f"After: {train_filter.count(True)} train, {test_easy_filter.count(True)} test_easy, {test_hard_filter.count(True)} test_hard")
        print("************************************")
        checked_dict[superclass]['train_filter'] = train_filter
        checked_dict[superclass]['test_easy_filter'] = test_easy_filter
        checked_dict[superclass]['test_hard_filter'] = test_hard_filter

# Save the filtered dictionary
with open(info_file, 'w') as file:
    yaml.dump(checked_dict, file)
    print(f"Saved filtered info to {info_file}")
