import argparse
import os
import re
import json
import torch
import yaml
import numpy as np
from PIL import Image
import cv2
import torchvision.transforms as transforms
from tqdm import tqdm
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# Load configuration from a YAML file
def load_config(config_file):
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

# Load and preprocess an image from the given path
def load_image(image_path, transform):
    image_pil = Image.open(image_path).convert("RGB")
    image_pil_size = image_pil.size
    image = transform(image_pil)
    return image_pil, image, image_pil_size

# Load and build the model using a checkpoint and a configuration file
def load_model(device, checkpoint):
    model_config_path = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()  # Set the model to evaluation mode
    return model

# Get phrases from the filtered logits in batch mode
def get_phrases_from_posmap_batch(logits_filt, tokenized, tokenizer, text_threshold):
    pred_phrases = []
    for logit in logits_filt:
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer)
        pred_phrases.append(pred_phrase)
    return pred_phrases

# Generate grounding outputs such as bounding boxes and phrases
def get_grounding_output(model, image, caption, box_threshold, text_threshold, device="cpu"):
    # Preprocess the caption
    caption = caption.lower().strip() + "."

    # Move the model and image to the specified device (CPU or GPU)
    model = model.to(device)
    image = image.to(device)

    # Forward pass through the model without calculating gradients
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])

    # Extract logits and bounding boxes
    logits = outputs["pred_logits"].sigmoid()[0]
    boxes = outputs["pred_boxes"][0]

    # Apply threshold to filter logits and boxes
    filt_mask = logits.max(dim=1)[0] > box_threshold
    logits_filt = logits[filt_mask]
    boxes_filt = boxes[filt_mask]

    # Tokenize the caption
    tokenized = model.tokenizer(caption)

    # Get phrases corresponding to the filtered logits
    pred_phrases = get_phrases_from_posmap_batch(logits_filt, tokenized, model.tokenizer, text_threshold)

    return boxes_filt, pred_phrases

# Extract the frame number from the filename using a regex pattern
def extract_image_number(filename):
    match = re.search(r'original_frame_(\d+)\.jpg', filename)
    if match:
        return int(match.group(1))
    return None

# Prepare the image for model input by applying transformations and converting to tensor
def prepare_image(image, transform, device):
    image = transform.apply_image(image)
    image = torch.as_tensor(image, device=device.device)
    return image.permute(2, 0, 1).contiguous()

# Save bounding boxes to a specified file
def save_boxes_to_file(file_name_without_ext, boxes, file_path='boxes.txt'):
    with open(file_path, 'a') as file:
        for box in boxes:
            box_str = ' '.join([f'{coord:.6f}' for coord in box])
            file.write(f'{file_name_without_ext} {box_str}\n')

# Main function to run the entire pipeline
def run_pipeline(config, start_from=None):
    # Determine the device (CPU or GPU) to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the DINO model
    dino_model = load_model(device=device, checkpoint=config["dino_checkpoint"])
    
    # Set the input and output directories
    base_path = config["root_imagenet"]
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)  # Create the output directory if it doesn't exist

    for list_file in config["image_lists"]:
    
        # Load image paths, labels, and prompts from the list file
        image_paths, labels, prompts = [], [], []
        with open(list_file, 'r') as file:
            for line in file:
                parts = line.strip().split()
                image_paths.append(os.path.join(base_path, parts[0]))
                labels.append(parts[1])
                prompts.append(parts[2])


        set_id = list_file.split('/')[-1].split('.')[0]
        os.makedirs(os.path.join(config['intermediate_file_dir'], set_id), exist_ok=True)
        box_filename = f"boxes_{config['data_id']}.txt"
        print(f"Processing images from {list_file} ...")
        print(f"Saving bounding boxes to {box_filename} ...\n")

        # Set the processing range
        part_start_idx = 0
        part_end_idx = len(image_paths)
        start_idx = start_from if start_from is not None else 0

        # Initialize the output file for saving bounding boxes
        open(os.path.join(config['intermediate_file_dir'], set_id, box_filename), 'w').close()

        # Set thresholds for box and text detection
        box_thresh = config['box_threshold']
        text_thresh = config['text_threshold']

        # Define image transformations
        transform = transforms.Compose([
            transforms.Resize((500, 500)),  # Resize images to a fixed size
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Process each image in the list
        for idx, image_path in enumerate(tqdm(image_paths)):
            _, image_dino, image_pil_size = load_image(image_path, transform)
            boxes, _ = get_grounding_output(dino_model, image_dino, prompts[idx], box_thresh, text_thresh, device)

            if boxes.nelement() == 0:  # Skip if no boxes are detected
                continue

            # Adjust the box coordinates to the original image size
            image = cv2.imread(image_path)
            scale_tensor = torch.tensor([image.shape[1], image.shape[0], image.shape[1], image.shape[0]], device=device)
            boxes = boxes * scale_tensor
            boxes[:, :2] -= boxes[:, 2:] / 2
            boxes[:, 2:] += boxes[:, :2]

            # Save the bounding boxes to the output file
            file_name_without_ext = os.path.splitext(os.path.basename(image_path))[0]
            save_boxes_to_file(file_name_without_ext, boxes, os.path.join(config['intermediate_file_dir'], set_id, box_filename))

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)

    parser.add_argument('--start_from', type=int, default=0, help='Starting batch index within the specified part')
    parser.add_argument('--dino_checkpoint', type=str, default='GroundingDINO/dino_weights/groundingdino_swint_ogc.pth',
                        help='Path to the DINO checkpoint')
    parser.add_argument('--root_imagenet', type=str, default='/media/data/Datasets/imagenet21k_resized',
                        help='Directory containing input images from ImageNet')
    parser.add_argument('--data_id', type=str, default='sood_imagenet', help='Identifier')
    parser.add_argument('--image_format', type=str, default='.JPEG',
                        help='Format of the input images')
    parser.add_argument('--image_lists', nargs='+', type=str, default=['lists/classification/test_easy_ood.txt',
                                        'lists/classification/test_hard_ood.txt'], help='List of image lists to extract masks')
    parser.add_argument('--box_threshold', type=float, default=0.3,
                        help='Threshold for bounding box predictions')
    parser.add_argument('--text_threshold', type=float, default=0.25,
                        help='Threshold for text predictions')

    args = parser.parse_args()

    config = vars(args)
    config['output_dir'] = f"{config['data_id']}_seg_pseudomasks_test"
    config['intermediate_file_dir']=f"seg_masks"
    os.makedirs(config['intermediate_file_dir'], exist_ok=True)

    print("Configuration:")
    for key, value in config.items():
        print(f"{key}: {value}")

    print("Intermediate files will be saved in:", config['intermediate_file_dir'], "\n")

    run_pipeline(config, config['start_from'])