import os
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import yaml
import argparse

def load_config(config_path):
    """Load the configuration file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def print_real_time_stats(stats):
    """Print real-time processing statistics."""
    total_processed = stats['level1'] + stats['level2'] + stats['skipped']
    level1_percentage = (stats['level1'] / total_processed) * 100 if total_processed > 0 else 0
    level2_percentage = (stats['level2'] / total_processed) * 100 if total_processed > 0 else 0
    skipped_percentage = (stats['skipped'] / total_processed) * 100 if total_processed > 0 else 0

    print(f"Processed: {total_processed}/{stats['total']}, "
          f"Selected: {level1_percentage:.2f}%, "
          f"Borderline: {level2_percentage:.2f}%, "
          f"Skipped: {skipped_percentage:.2f}%")

def on_key(event, fig, image_paths, info_lines, config, stats):
    """Handle key press events during image processing."""
    current_index = stats['current_index']
    mask_path = image_paths[current_index].strip()
    selected_info = info_lines[current_index].strip()
    borderline_info = info_lines[current_index].strip()

    if event.key == '1':  # Save as level 1
        with open(config['output_file_level1'], 'a') as f:
            f.write(f"{mask_path}\n")
        with open(config['output_image_level1'], 'a') as f:
            f.write(f"{selected_info}\n")
        stats['level1'] += 1
        print(f"Selected as [ACCEPTED] and saved: {mask_path}")
    elif event.key == '2':  # Save as borderline
        with open(config['output_file_level2'], 'a') as f:
            f.write(f"{mask_path}\n")
        with open(config['output_image_level2'], 'a') as f:
            f.write(f"{borderline_info}\n")
        stats['level2'] += 1
        print(f"Selected as [BORDERLINE] and saved: {mask_path}")
    elif event.key == '3':  # Skip the image
        stats['skipped'] += 1
        print(f"DISCARDED: {mask_path}")
    else:
        return  # Do nothing if another key is pressed

    stats['current_index'] += 1  # Update the current index
    with open(config['stat_file'], 'w') as json_file:
        json.dump(stats, json_file)
    
    print_real_time_stats(stats)
    update_plot(fig, image_paths, info_lines, stats, config)

def update_plot(fig, image_paths, info_lines, stats, config):
    """Update the plot with the current image and its mask."""
    current_index = stats['current_index']
    if current_index >= len(image_paths):
        print("All images processed.")
        plt.close(fig)
        return

    mask_path = image_paths[current_index].strip()
    image_path = mask_path.replace('_mask.png', '.JPEG')

    image_path = os.path.join(config['root_imagenet'], config['image_folder'], image_path)
    mask_path = os.path.join(config['mask_folder'], mask_path)

    # original_path = mask_path.replace(
    #     "/media/data2/imagenet21k_masks/output_test_hard_sam2/",
    #     "/media/data/Datasets/imagenet21k_resized/imagenet21k_train/"
    # ).replace("_mask.png", ".JPEG")
    
    mask_image = mpimg.imread(mask_path)
    original_image = mpimg.imread(image_path)

    axes[0].cla()
    axes[0].imshow(mask_image)
    axes[0].set_title('Mask Image')
    axes[0].axis('off')

    axes[1].cla()
    axes[1].imshow(original_image)
    axes[1].set_title('Original Image')
    axes[1].axis('off')

    info_parts = info_lines[current_index].split()
    prompt = info_parts[2]
    fig.suptitle(f"Processing image with prompt: {prompt}")

    fig.canvas.draw_idle()

def plot_and_select_images(config, start_index=0):
    """Main function to plot images and allow user selection."""
    if start_index == 0:
        for file_key in ['output_file_level1', 'output_file_level2', 'output_image_level1', 'output_image_level2', 'final_output_file']:
            open(config[file_key], 'w').close()

    stats = {'total': 0, 'level1': 0, 'level2': 0, 'skipped': 0, 'current_index': start_index}

    with open(config['input_file_path'], 'r') as f:
        image_paths = f.readlines()
        stats['total'] = len(image_paths)

    with open(config['info_file_path'], 'r') as f:
        info_lines = f.readlines()

    if os.path.exists(config['stat_file']):
        with open(config['stat_file'], 'r') as json_file:
            stats = json.load(json_file)

    global axes
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    update_plot(fig, image_paths, info_lines, stats, config)
    
    fig.canvas.mpl_connect('key_press_event', lambda event: on_key(event, fig, image_paths, info_lines, config, stats))

    plt.show()

def find_start_index(info_file_path, output_file_level1, output_file_level2):
    """Find the index to start processing from, based on previous selections."""
    last_level1, last_level2 = "", ""

    if os.path.exists(output_file_level1):
        with open(output_file_level1, 'r') as f:
            level1_lines = f.readlines()
            if level1_lines:
                last_level1 = level1_lines[-1].strip()
    
    if os.path.exists(output_file_level2):
        with open(output_file_level2, 'r') as f:
            level2_lines = f.readlines()
            if level2_lines:
                last_level2 = level2_lines[-1].strip()

    start_index = 0
    with open(info_file_path, 'r') as f:
        info_lines = f.readlines()
        for i, line in enumerate(info_lines):
            if last_level1 in line or last_level2 in line:
                start_index = i + 1

    return start_index

def main():
    parser = argparse.ArgumentParser(description="Process images and select levels using keypresses.")

    parser.add_argument('--root_imagenet', type=str, default='/media/data/Datasets/imagenet21k_resized',
                        help='Directory containing input images from ImageNet')
    parser.add_argument('--data_id', type=str, default='sood_imagenet', help='Identifier') #sood_imagenet

    args = parser.parse_args()
    config = vars(args)

    config['intermediate_file_dir'] = f"seg_masks"

    print("Configuration:")
    for key, value in config.items():
        print(f"{key}: {value}")

    # list of subfolders
    for folder in os.listdir(config['intermediate_file_dir']):

        mask_file = os.path.join(config['intermediate_file_dir'], folder, f"mask_processed_sam2_{config['data_id']}.txt")
        mask_folder = os.path.join(config['root_imagenet'], f"{config['data_id']}_seg_pseudomasks_{folder}_sam2")

        config['mask_folder'] = mask_folder
        config['image_folder'] = 'imagenet21k_train'

        # INPUT
        # input_file_path: "/media/data2/imagenet21k_masks/output_test_hard_sam2/mask_processed_sam2_sood_imagenet.txt"
        config['input_file_path'] = mask_file
        print(f"\nReading from {mask_file}")
        if not os.path.exists(mask_file):
            raise FileNotFoundError(f"File not found: {mask_file}")

        image_file = os.path.join("lists", "classification", f"{folder}_{config['data_id']}.txt")
        config['info_file_path'] = image_file

        # OUTPUT
        # output_file_level1: "/media/data2/imagenet21k_masks/output_test_hard_sam2/mask_selected.txt"
        config['output_file_level1'] = os.path.join(config['intermediate_file_dir'], folder, f"mask_selected_{config['data_id']}.txt")
        # output_file_level2: "/media/data2/imagenet21k_masks/output_test_hard_sam2/mask_borderline.txt"
        config['output_file_level2'] = os.path.join(config['intermediate_file_dir'], folder, f"mask_borderline_{config['data_id']}.txt")
        # output_image_level1: "/media/data2/imagenet21k_masks/output_test_hard_sam2/image_selected.txt"
        config['output_image_level1'] = os.path.join(config['intermediate_file_dir'], folder, f"image_selected_{config['data_id']}.txt")
        # output_image_level2: "/media/data2/imagenet21k_masks/output_test_hard_sam2/image_borderline.txt"
        config['output_image_level2'] = os.path.join(config['intermediate_file_dir'], folder, f"image_borderline_{config['data_id']}.txt")

        #output test list
        # final_output_file: "/media/data2/imagenet21k_masks/output_test_hard_sam2/final_test_file.txt"
        config['final_output_file'] = os.path.join("lists", "segmentation", f"{folder}_{config['data_id']}_sam2.txt")

        stat_file = os.path.join(config['intermediate_file_dir'], folder, f"stats_{config['data_id']}.json")
        config['stat_file'] = stat_file
        if os.path.exists(stat_file):
            with open(stat_file, 'r') as json_file:
                stats = json.load(json_file)
                start_index = stats.get('current_index', 0)
                print(f"Loaded stats from {stat_file}")
                print(f"Resuming from index: {start_index}")
        else:
            start_index = 0

        print("+++++ USAGE INSTRUCTION:")
        print("This tool allow to manually rank and discard images and the generated masks.")
        print("Press [1] to select as [ACCEPTED], [2] to select as [BORDERLINE], [3] to [DISCARD] the image.")
        print("Press [Q] to quit the tool.")

        plot_and_select_images(config, start_index)

if __name__ == "__main__":
    main()
