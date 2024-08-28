import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import random
import os
import argparse
from tqdm import tqdm

# Function to calculate statistics
def calculate_statistics(data):
    stats = {}
    for category, items in tqdm(data.items(), desc='Calculating statistics'):
        df = pd.DataFrame(items)
        for prompt in df['prompt_specific'].unique():
            prompt_df = df[df['prompt_specific'] == prompt]
            mean = prompt_df['normalized_logit'].mean()
            std = prompt_df['normalized_logit'].std()

            # Using 1.5 * IQR rule to detect outliers
            Q1 = prompt_df['normalized_logit'].quantile(0.25)
            Q3 = prompt_df['normalized_logit'].quantile(0.75)
            IQR = Q3 - Q1
            outliers = prompt_df[(prompt_df['normalized_logit'] < (Q1 - 1.5 * IQR)) | (prompt_df['normalized_logit'] > (Q3 + 1.5 * IQR))]
            
            stats.setdefault(category, []).append({
                'prompt_specific': prompt,
                'folder_name': os.path.basename(os.path.dirname(prompt_df['image_path'].values[0])),
                'mean_normalized_logit': mean,
                'std_normalized_logit': std,
                'outliers': outliers.to_dict(orient='records')
            })
    return stats


# Function to save stats to JSON
def save_stats_to_json(stats, output_path):
    json_data = {}
    for category, stat_list in tqdm(stats.items(), desc='Saving statistics to JSON'):
        sorted_stat_list = sorted(stat_list, key=lambda x: x['mean_normalized_logit'])
        json_data[category] = [{
            'prompt_specific': stat['prompt_specific'],
            'folder_name': stat['folder_name'],
            'mean_normalized_logit': stat['mean_normalized_logit'],
            'std_normalized_logit': stat['std_normalized_logit'],
            'outliers': [{'image_path': outlier['image_path'], 'normalized_logit': outlier['normalized_logit']} for outlier in stat['outliers']]
        } for stat in sorted_stat_list]
    with open(output_path, 'w') as f:
        json.dump(json_data, f, indent=4)

# Main function
def main():

    parser = argparse.ArgumentParser(description='Calculate statistics for CLIP model outputs')
    parser.add_argument('--data', type=str, default='logit_dict_total_test.json', help='Path to input JSON data file')
    parser.add_argument('--output_list', type=str, default='statistics.json', help='Path to output JSON data file')

    folder = "scoring"
    if not os.path.exists(folder):
        raise Exception(f"Folder {folder} does not exist")
    
    # Load JSON data
    args = parser.parse_args()
    with open(os.path.join(folder, args.data), 'r') as f:
        data = json.load(f)
    
    # Calculate statistics
    stats = calculate_statistics(data)

    # Save stats to JSON
    save_stats_to_json(stats, os.path.join(folder, args.output_list))

if __name__ == "__main__":
    main()

