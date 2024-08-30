import os
import numpy as np
import cv2
import torch
import argparse
import yaml
from tqdm import tqdm
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

def load_config(config_path):
    """Load the configuration file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def save_mask_processed(file_name, file_path='mask_processed.txt'):
    """Append the name of the processed mask to a file."""
    with open(file_path, 'a') as file:
        file.write(f'{file_name}\n')

def draw_bboxes(image, bboxes):
    """Draw bounding boxes on the image."""
    for bbox in bboxes:
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return image

def save_mask_mapping(mask_list, output_dir, base_name, mask_value):
    """Save the generated mask to the specified directory."""
    mask_shape = mask_list.shape[-2:]
    mask_img = np.zeros(mask_shape)
    mask_np = mask_list
    for idx, mask in enumerate(mask_np):
        mask_img[mask > 0] = int(mask_value)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{base_name}_mask.png')
    cv2.imwrite(output_path, mask_img)

def process_images(config):
    """Main function to process images and save masks."""
    # Initialize SAM2 model
    predictor = SAM2ImagePredictor(build_sam2(config['model_cfg'], config['checkpoint']))

    debug_cont = 0

    # Load image and bounding box information
    images = {}
    with open(config['image_info_file'], 'r') as file:
        for line in tqdm(file, desc='Loading image information'):
            parts = line.strip().split()
            image_path = os.path.join(config['root_imagenet'], parts[0])
            mask_value = int(parts[1]) + 1  # Avoid 0 values (background)
            if image_path not in images:
                images[image_path] = {'bboxes': [], 'mask_value': mask_value}

    with open(config['input_file'], 'r') as file:
        for line in tqdm(file.readlines(), desc='Loading bounding boxes'):
            parts = line.strip().split()
            image_name = parts[0]
            bbox = [float(parts[i]) for i in range(1, 5)]
            for image_path in images:
                if image_name in image_path:
                    images[image_path]['bboxes'].append(bbox)
                    break

    # Process each image
    for image_path, data in tqdm(images.items(), desc='Processing images'):
        if os.path.exists(image_path):
            image = Image.open(image_path).convert("RGB")
            image_np = np.array(image)
            image_with_bboxes = draw_bboxes(image_np.copy(), data['bboxes'])

            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                predictor.set_image(image_np)
                boxes = np.array([np.array(bbox) for bbox in data['bboxes']])
                if boxes.shape[0] == 0:
                    continue

                masks, _, _ = predictor.predict(point_coords=None,
                                                point_labels=None,
                                                box=boxes,
                                                multimask_output=False)

            combined_mask = np.zeros((masks.shape[-2], masks.shape[-1]))
            if masks.ndim == 3:
                final_mask = masks
            else:
                for j in range(masks.shape[0]):
                    combined_mask = np.maximum(combined_mask, masks[j])
                final_mask = combined_mask

            final_mask = final_mask.astype(bool)
            base_name = os.path.basename(image_path).split('.')[0]
            folder_name = os.path.basename(os.path.dirname(image_path))
            save_mask_mapping(final_mask, os.path.join(config['output_dir'], folder_name), base_name, data['mask_value'])
            save_mask_processed(os.path.join(folder_name, f'{base_name}_mask.png'), config['mask_processed_file'])
            debug_cont += 1
        else:
            print(f"Image not found: {image_path}")

    print(f"Processed {debug_cont} images")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images and save masks using SAM2.")

    parser.add_argument('--image_lists', nargs='+', type=str, default=['lists/classification/test_easy_ood.txt',
                                                                       'lists/classification/test_hard_ood.txt'],
                        help='List of image lists to extract masks')
    parser.add_argument('--root_imagenet', type=str, default='/media/data/Datasets/imagenet21k_resized',
                        help='Directory containing input images from ImageNet')

    parser.add_argument('--data_id', type=str, default='sood_imagenet', help='Identifier')

    parser.add_argument('--model_cfg', type=str, default='../../segment-anything-2/sam2_configs/sam2_hiera_l.yaml',
                        help='Path to the model configuration file')

    parser.add_argument('--checkpoint', type=str, default='segment-anything-2/checkpoints/sam2_hiera_large.pt',
                        help='Path to the model checkpoint file')

    args = parser.parse_args()

    # Load configuration and run the pipeline
    config = vars(args)
    config['intermediate_file_dir'] = f"seg_masks"


    print("Configuration:")
    for key, value in config.items():
        print(f"{key}: {value}")

    print("Intermediate files will be saved in:", config['intermediate_file_dir'], "\n")

    for list_file in config['image_lists']:
        set_id = list_file.split('/')[-1].split('.')[0]
        box_filename = os.path.join(set_id, f"boxes_{config['data_id']}.txt")
        config['input_file'] = os.path.join(config['intermediate_file_dir'], box_filename)
        config['image_info_file'] = list_file
        config['mask_processed_file'] = os.path.join(config['intermediate_file_dir'], set_id, f"mask_processed_sam2_{config['data_id']}.txt")
        config['output_dir'] = os.path.join(config['root_imagenet'], f"{config['data_id']}_seg_pseudomasks_{set_id}_sam2")
        os.makedirs(config['output_dir'], exist_ok=True)

        print(f"Processing images from {list_file}, boxes from {config['input_file']}")
        print(f"Saving intermediate file {config['mask_processed_file']}")
        print(f"Saving masks to {config['output_dir']}\n")

        process_images(config)
