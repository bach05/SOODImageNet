import json
import os
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import argparse

# Function to generate image lists divided by percentiles
def generate_image_lists_by_percentiles(json_file, base_path, output_files, p_value_1, p_value_2):
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Dictionaries to keep track of image information for each subset
    subsets = {
        'train': defaultdict(list),
        'test_easy': defaultdict(list),
        'test_hard': defaultdict(list)
    }

    # Iterate through the keys of the dictionary
    for class_index, (class_name, class_data) in tqdm(enumerate(data.items()), desc='Generating image lists by class', total=len(data)):
        # List to keep track of image information for each class
        images_info = []

        for subclass in class_data:
            prompt_specific = subclass['prompt_specific']
            folder_name = subclass['folder_name']
            mean_normalized_logit = subclass['mean_normalized_logit']
            outliers_paths = {outlier['image_path'] for outlier in subclass['outliers']}
            
            folder_path = os.path.join(base_path, folder_name)
            
            # Iterate through the images in the specified folder
            for root, _, files in os.walk(folder_path):
                for file in files:
                    image_path = os.path.join(root, file)
                    if image_path not in outliers_paths:
                        relative_image_path = os.path.relpath(image_path, base_path)
                        images_info.append((f"imagenet21k_train/{relative_image_path}", class_index, class_name, prompt_specific, mean_normalized_logit))

        # Convert the list to a numpy array to facilitate percentile operations
        images_info = np.array(images_info, dtype=object)
        
        # Calculate the percentiles for the current class
        scores = images_info[:, 4].astype(float)
        p40 = np.percentile(scores, p_value_1)
        p20 = np.percentile(scores, p_value_2)

        # Divide the images into three files based on the calculated percentiles for the current class
        for image_info in images_info:
            image_path, class_index, class_name, prompt_specific, score = image_info
            score = float(score)
            if score >= p40:
                subsets['train'][prompt_specific].append(image_info)
            elif score >= p20:
                subsets['test_easy'][prompt_specific].append(image_info)
            else:
                subsets['test_hard'][prompt_specific].append(image_info)

    # Write the images to the respective output files
    with open(output_files['train'], 'w') as train_file, \
         open(output_files['test_easy'], 'w') as test_easy_file, \
         open(output_files['test_hard'], 'w') as test_hard_file:

        for subset_name, subset_data in subsets.items():
            for prompt_specific, image_infos in subset_data.items():
                for image_info in image_infos:
                    image_path, class_index, class_name, prompt_specific, score = image_info
                    if subset_name == 'train':
                        train_file.write(f"{image_path} {class_index} {class_name} {prompt_specific}\n")
                    elif subset_name == 'test_easy':
                        test_easy_file.write(f"{image_path} {class_index} {class_name} {prompt_specific}\n")
                    else:
                        test_hard_file.write(f"{image_path} {class_index} {class_name} {prompt_specific}\n")

def main():
    parser = argparse.ArgumentParser(description="Generate image lists divided by percentiles")
    parser.add_argument('--json_file', type=str, default='./scoring/statistics.json', help='Path to the JSON file with statistic from outliers_detection.py')
    parser.add_argument('--root_imagenet', type=str, default='/media/data/Datasets/imagenet21k_resized/imagenet21k_train/', help='Base path for the images')
    parser.add_argument('--data_id', default='selected2imagenet', type=str, help='Data ID for the dataset')
    parser.add_argument('--p_value_1', type=float, default=40, help='Percentile value for the training set (default: 40)')
    parser.add_argument('--p_value_2', type=float, default=20, help='Percentile value for the easy test set (default: 20)')
    
    args = parser.parse_args()

    output_files = {
        'train': os.path.join('lists', 'classification', f"train_iid_{args.data_id}.txt"),
        'test_easy': os.path.join('lists', 'classification', f"test_easy_ood_{args.data_id}.txt"),
        'test_hard': os.path.join('lists', 'classification', f"test_hard_ood_{args.data_id}.txt")
    }

    # Generate the image lists
    generate_image_lists_by_percentiles(args.json_file, args.root_imagenet, output_files, args.p_value_1, args.p_value_2)

if __name__ == "__main__":
    main()
