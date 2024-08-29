import sys

sys.path.append("..")
from tqdm import tqdm
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import yaml
from transformers import  CLIPSegProcessor, CLIPSegForImageSegmentation
from PIL import Image
from torch.utils.data import DataLoader
import transformers
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from torch.utils.data import Dataset

def custom_collate_fn(batch):

    images = [el[0] for el in batch]
    text_inputs = [el[1] for el in batch]
    image_paths = [el[2] for el in batch]
    class_ids = [el[3] for el in batch]
    super_names = [el[4] for el in batch]
    sub_names = [el[5] for el in batch]

    return images, text_inputs, image_paths, class_ids, super_names, sub_names

def process_line(line, data_base_path):
    image_path, class_id, superclass_name, subclass_name = line.strip().split()
    full_image_path = os.path.join(data_base_path, image_path)
    image_np = cv2.imread(full_image_path)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image_np)
    return image, f"{superclass_name}", image_path, int(class_id), superclass_name, subclass_name

def process_batch(batch_lines, data_base_path, processor, device):
    images = []
    text_inputs = []
    image_paths = []
    class_ids = []
    super_names = []
    sub_names = []

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_line, line, data_base_path) for line in batch_lines]
        for future in as_completed(futures):
            image, text_input, image_path, class_id, super_name, sub_name = future.result()
            images.append(image)
            text_inputs.append(text_input)
            image_paths.append(image_path)
            class_ids.append(class_id)
            super_names.append(super_name)
            sub_names.append(sub_name)

    inputs = processor(text=text_inputs, images=images, padding="max_length", return_tensors="pt").to(device)
    return inputs, images, image_paths, class_ids, super_names, sub_names


class BasicDataset(Dataset):
    def __init__(self, data_list, image_root):

        self.data_list = data_list
        self.image_root = image_root

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        image_path, class_id, superclass_name, subclass_name = self.data_list[idx].strip().split(' ')
        full_image_path = os.path.join(self.image_root, image_path)
        image_np = cv2.imread(full_image_path)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image_np)

        return image, f"{superclass_name}", image_path, int(class_id), superclass_name, subclass_name

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Label with CLIPSeg model')
    # Adding fields from the config file as command-line arguments
    parser.add_argument('--root_imagenet', type=str, default='/media/data/Datasets/imagenet21k_resized', help='Path to the base directory of the dataset')
    parser.add_argument('--image_lists', nargs='+', type=str, default=['lists/classification/train_iid.txt'], help='List of image lists to extract masks')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for processing images')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker threads for data loading')
    parser.add_argument('--model', type=str, default='clipseg', help='Model to use for labeling (e.g., clipseg)')
    parser.add_argument('--data_id', type=str, default='sood_imagenet', help='Folder to save the output pseudomasks')
    parser.add_argument('--vis_images', type=bool, default=False, help='Flag to visualize images during processing')

    args = parser.parse_args()

    config = vars(args)
    config['output_folder'] = f"{config['data_id']}_seg_pseudomasks"

    print("Configuration:")
    for key, value in config.items():
        print(f"{key}: {value}")

    print(f"\n>>>>> Segmentation masks will be saved in {os.path.join(config['root_imagenet'], config['output_folder'])}")

    proceed = input("\nDo you want to proceed with the above configurations? (y/n): ").lower()
    if proceed != 'y':
        print("Exiting...")
        exit()

    # disable some warnings
    transformers.logging.set_verbosity_error()
    transformers.logging.disable_progress_bar()
    warnings.filterwarnings('ignore')

    out_folder = config['output_folder'] + f"_{config['model']}"

    batch_size = config['batch_size']

    # INIT FLORENCE MODEL
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").eval().to(device)

    # INIT IMAGE LIST
    root_imagenet = config['root_imagenet']
    image_lists = config['image_lists']
    task_prompt = '<REFERRING_EXPRESSION_SEGMENTATION>'

    for image_list in image_lists:
        with open(image_list, 'r') as f:
            lines = f.readlines()

        # pass lines to the dataloader
        dataset = BasicDataset(lines, image_root=root_imagenet)
        dataloader = DataLoader(dataset, batch_size=batch_size,
                                shuffle=False,
                                collate_fn=custom_collate_fn,
                                num_workers=config['num_workers'])

        out_list = os.path.join("lists", "segmentation" , os.path.basename(image_list).replace(".txt", f"_{config['model']}.txt"))
        print(f"Labeling images in {image_list}")
        print(f"Saving pseudo masks in {out_list}")
        time.sleep(0.5)

        out_file_list = []

        for i, (images, text_inputs, image_paths, class_ids, superclass_names, subclass_names) in enumerate(tqdm(dataloader, desc='Labeling images')):

            inputs = processor(text=text_inputs, images=images, padding="max_length", return_tensors="pt").to(device)

            # predict
            with torch.no_grad():
                outputs = model(**inputs)

            preds = outputs.logits.unsqueeze(1)

            pred_list = [torch.sigmoid(preds[i][0]).cpu().numpy() for i in range(preds.shape[0])]

            for image, mask, class_id, image_path, superclass_name, subclass_name in zip(images, pred_list, class_ids, image_paths, superclass_names, subclass_names):

                mask = (mask > 0.2*mask.max()).astype(np.uint8) * (class_id+1)
                mask = cv2.resize(mask, (image.width, image.height), interpolation=cv2.INTER_NEAREST)

                # Save image with the same structure as input
                folder, image_file = os.path.split(image_path)
                mask_folder = os.path.join(root_imagenet, out_folder, folder)
                if not os.path.exists(mask_folder):
                    os.makedirs(mask_folder, exist_ok=True)
                mask_file = image_file.replace('.JPEG', '.png')
                cv2.imwrite(os.path.join(mask_folder, mask_file), mask)

                out_file_list += [f"{image_path} {os.path.join(out_folder, folder, mask_file)} {class_id+1} {superclass_name} {subclass_name}\n"]

                if config['vis_images']:
                    # Subplots
                    fig, ax = plt.subplots(1, 2, figsize=(20, 20))
                    ax[0].imshow(image)
                    ax[0].set_title('Original Image')
                    ax[1].imshow(mask)
                    ax[1].set_title('Pseudo Mask')
                    plt.show()
                    plt.close()

            # Update list file
            if i % 100*batch_size == 0:
                with open(out_list, 'w') as f:
                    f.writelines(out_file_list)




