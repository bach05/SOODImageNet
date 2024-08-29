import yaml
import os
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Process some IMAGEMET.')
parser.add_argument('--data_id', type=str, default="selected2imagenet", help='data id')
parser.add_argument('--min_num_subclasses', type=int, default=10, help='Minimum number of subclasses')

args = parser.parse_args()

# Read the existing yaml file
data_id = args.data_id
min_num_subclasses = args.min_num_subclasses
data_file = f"mapping/{data_id}_sub_{min_num_subclasses}.yaml"
with open(data_file, 'r') as file:
    data_dict = yaml.safe_load(file)

info_file = f"mapping/{data_id}_sub_{min_num_subclasses}_equalized.yaml"
if os.path.exists(info_file):
    with open(info_file, 'r') as file:
        cmd = input(f"Equalized file {info_file} exists, do you want to load it? y/n \n")
        if cmd == 'n':
            print("OVERWRITING the equalized file")
            checked_dict = {}
        else:
            checked_dict = yaml.safe_load(file)
            print(f"Loaded Equalized info from {info_file}")
else:
    print(f"Equalized file {info_file} does not exist, creating a new one")
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

        #generate random bool values to keep or discard from original list
        indexes = np.random.choice(num_subclasses, th_max_subclasses, replace=False)
        bool_indexes = [True if i in indexes else False for i in range(num_subclasses)]

        checked_dict[superclass]['equalization_filter'] = bool_indexes
    else:
        checked_dict[superclass]['equalization_filter'] = [True for i in range(num_subclasses)]

# Save the filtered dictionary
with open(info_file, 'w') as file:
    yaml.dump(checked_dict, file)
    print(f"Saved filtered info to {info_file}")
