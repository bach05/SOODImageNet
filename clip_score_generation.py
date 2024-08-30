import torch
import clip
from PIL import Image
import numpy as np
from tqdm import tqdm
import json
import argparse
from torch.utils.data import DataLoader, Dataset
import yaml
import os

def print_config(config):
    print("Configurations:")
    for key, value in vars(config).items():
        print(f"{key}: {value}")

# Custom dataset for loading and preprocessing images and prompts
class ImagePromptDataset(Dataset):
    def __init__(self, image_paths, prompts, specific_prompts, preprocess):
        self.image_paths = image_paths
        self.prompts = prompts
        self.specific_prompts = specific_prompts
        self.preprocess = preprocess

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = self.preprocess(Image.open(self.image_paths[idx]))
        prompt = clip.tokenize([self.prompts[idx]])[0]
        return image, prompt

def read_file(file_path):
    images = []
    prompts_general = []
    prompts_specific = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split(" ")
            image_path = parts[0]
            prompt_general = " ".join(parts[2:3])
            prompt_specific = " ".join(parts[3:])
            images.append(image_path)
            prompts_general.append(prompt_general)
            prompts_specific.append(prompt_specific)
    return images, prompts_general, prompts_specific


def clip_score_generation(config):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    data_root = config.root_imagenet
    out_folder = "scoring"
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
        print(f"Created folder {out_folder}")
    output_file_path = os.path.join(out_folder, config.output_file)

    all_image_paths = []
    all_prompts_general = []
    all_prompts_specific = []

    mapping_file = f'mapping/{config.data_id}_sub_{config.min_num_subclasses}_equalized.yaml'
    with open(mapping_file, 'r') as file:
        mapping = yaml.safe_load(file)

    for superclass, info in tqdm(mapping.items(), desc='Loading images and prompts'):

        folders = info['folders']
        subclasses = info['classes']
        if info.get('equalization_filter') is not None:
            equalization_filter = info['equalization_filter']
            folders = [folder for i, folder in enumerate(folders) if equalization_filter[i]]
            subclasses = [subclass for i, subclass in enumerate(subclasses) if equalization_filter[i]]

        for folder, subclass in zip(folders, subclasses):
            image_list = os.listdir(os.path.join(data_root, folder))
            for image in image_list:
                all_image_paths.append(os.path.join(data_root, folder, image))
                all_prompts_general.append(superclass)
                all_prompts_specific.append(subclass)

    dataset = ImagePromptDataset(all_image_paths, all_prompts_general, all_prompts_specific, preprocess)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, num_workers=16, pin_memory=True)

    logit_dict = {}
    counter = 0

    # Process images and prompts with CLIP model
    with torch.no_grad():
        for images, texts in tqdm(dataloader, desc='Generating scores'):
            images = images.to(device)
            texts = texts.to(device)
            
            image_features = model.encode_image(images)
            text_features = model.encode_text(texts)
            
            logits_per_image, logits_per_text = model(images, texts)
            
            for i in range(images.size(0)):
                image_path = all_image_paths[counter]
                prompt_general = all_prompts_general[counter]
                prompt_specific = all_prompts_specific[counter]
                
                logit_value = logits_per_image[i][i].item()
                if prompt_general not in logit_dict:
                    logit_dict[prompt_general] = []
                logit_dict[prompt_general].append({'image_path': image_path, 'logit': logit_value, 'prompt_specific': prompt_specific})
                counter += 1

    # Normalize logit values
    for prompt_general in logit_dict:
        logits = [entry['logit'] for entry in logit_dict[prompt_general]]
        logit_min = min(logits)
        logit_max = max(logits)
        for entry in logit_dict[prompt_general]:
            entry['normalized_logit'] = (entry['logit'] - logit_min) / (logit_max - logit_min)

    with open(output_file_path, 'w') as json_file:
        json.dump(logit_dict, json_file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Clip Score Generation", add_help=True)

    # Existing argument

    # New arguments
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for processing')
    parser.add_argument('--data_id', type=str, default='selected2imagenet', help='Data id')
    parser.add_argument('--min_num_subclasses', type=int, default=10, help='Minimum number of subclasses')
    parser.add_argument('--root_imagenet', type=str, default='/media/data/Datasets/imagenet21k_resized',
                        help='Root directory of the dataset')
    parser.add_argument('--output_file', type=str, default='logit_dict_total_test.json',
                        help='Output file to save the results')

    args = parser.parse_args()
    print_config(args)

    clip_score_generation(args)