import os
import yaml
import matplotlib.pyplot as plt
import random
import numpy as np
import cv2
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Process some IMAGEMET.')
parser.add_argument('--data_id', type=str, default="selected2imagenet", help='data id')
parser.add_argument('--save_images', type=bool, default=False, help='save images')
args = parser.parse_args()


# Read the existing yaml file
data_id = args.data_id
data_file = f"mapping/{data_id}_out_dict.yaml"
with open(data_file, 'r') as file:
    data = yaml.safe_load(file)

info_file = f"mapping/{data_id}_human_checked_dict.yaml"
if os.path.exists(info_file):
    with open(info_file, 'r') as file:
        cmd = input(f"Checked file {info_file} exists, do you want to load it and continue labelling? y/n \n")
        if cmd == 'n':
            print("OVERWRITING the checked file")
            checked_dict = {}
        else:
            checked_dict = yaml.safe_load(file)
            print(f"Loaded checked info from {info_file}, continue labelling from here")
else:
    print(f"Checked file {info_file} does not exist, creating a new one")
    checked_dict = {}

superclasses_already_checked = list(checked_dict.keys())

# Process the data
samples_size = 16
for idx, (superclass, info) in enumerate(data.items()):

    if superclass in superclasses_already_checked:
        print(f"Skipping {superclass} as it was already checked.")
        continue

    # visualize an start screen with a text
    start_screen = np.zeros((1080, 1920, 3), dtype=np.uint8)
    cv2.putText(start_screen, f'Starting processing {superclass}', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2,
                (255, 255, 255), 2)
    cv2.putText(start_screen, 'INSTRUCTIONS: Press "k" to keep the subclass or "d" to discard', (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
    cv2.putText(start_screen, 'Press "q" to quit', (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
    cv2.putText(start_screen, 'Press any key to continue', (100, 400), cv2.FONT_HERSHEY_SIMPLEX, 2,(255, 255, 255), 2)
    cv2.putText(start_screen, f'Status: {idx}/{len(data)}', (100, 600), cv2.FONT_HERSHEY_SIMPLEX, 2,(255, 255, 255), 2)
    cv2.imshow('Start Screen', start_screen)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()

    if key == ord('q'):
        print("Quitting the process")
        break

    subclasses = info['classes']
    subclass_path = info['folders_paths']
    vis_check = info['vis_checks']

    manually_check_images = []

    checked_subclasses = np.array(subclasses)[vis_check].tolist()
    checked_subclass_path = np.array(subclass_path)[vis_check].tolist()

    for subclass, path in tqdm(zip(checked_subclasses, checked_subclass_path), total=len(checked_subclasses)):
        sample_images = random.sample(os.listdir(path), samples_size)

        # Create a grid of images
        rows, cols = 4, 4  # Adjust this for different grid dimensions
        grid_size = (256, 256)  # Resize images to fit in the grid
        grid_image = np.zeros((rows * grid_size[0], cols * grid_size[1], 3), dtype=np.uint8)

        for idx, image in enumerate(sample_images):
            img_path = os.path.join(path, image)
            img = cv2.imread(img_path)

            if img is None:
                continue

            img = cv2.resize(img, grid_size)  # Resize images to fit in the grid
            row = idx // cols
            col = idx % cols
            grid_image[row * grid_size[0]:(row + 1) * grid_size[0], col * grid_size[1]:(col + 1) * grid_size[1],
            :] = img

        # Display the grid
        cv2.imshow(f'{superclass} - {subclass}', grid_image)

        while True:
            key = cv2.waitKey(0)
            if key == ord('k'):
                manually_check_images.append(True)
                break
            elif key == ord('d'):
                manually_check_images.append(False)
                break
            else:
                print("Press 'k' to keep the subclass or 'd' to discard.")

        cv2.destroyAllWindows()

    info['manually_check_images'] = manually_check_images
    checked_dict[superclass] = info
    #save info
    with open(info_file, 'w') as file:
        yaml.dump(checked_dict, file)
        print(f"Saved checked dict into {info_file}")

    print("Finished processing", superclass)
