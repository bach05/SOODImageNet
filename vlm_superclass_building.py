import torch
import yaml
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import os
import pprint
from collections import deque
import random
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from PIL import Image
import math
import  argparse

#TO DO: 
# MANAGE THE OVERLAPPING OF CLASSES
# SAVE ALL THE IMAGES, splitted in sub_images eventually
# IF VIS CHECKS --> color, ELSE --> B&W

######### UTILS

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model from HuggingFace Hub
match_model = SentenceTransformer('all-mpnet-base-v2')
match_model = match_model.to(device)

#Load the model and processor
model_path = "./hf_models/paligemma-3b-mix-224"
gemma_model = PaliGemmaForConditionalGeneration.from_pretrained(model_path, local_files_only=True, device_map=device)
gemma_processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, device_map=device)

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def findBestMatch(label, target_label_list, confidence_threshold=0.5, match_query=None):

    label = label.replace("_", " ")

    query = (f"{label}")
    passage = target_label_list

    query_embedding = match_model.encode(query)
    passage_embeddings = match_model.encode(passage)

    similarity = match_model.similarity(query_embedding, passage_embeddings)

    sentence_similarity = similarity.tolist()[0]

    #create dict
    sentence_similarity = [(sentence, similarity) for sentence, similarity in zip(passage, sentence_similarity)]

    # Sort the list in descending order of similarity score
    sentence_similarity.sort(key=lambda x: x[1], reverse=True)
    sentence_similarity_np = np.array(sentence_similarity)

    # top k highest scores
    top_k = [(sentence, similarity) for sentence, similarity in sentence_similarity if similarity > confidence_threshold]
    return top_k

def rank_label(label, target_labels):

    label_ = label.replace("_", " ")
    target_label_list = [target_label.replace("_", " ") for target_label in target_labels]

    query = (f"{label_}")
    passage = target_label_list

    query_embedding = match_model.encode(query)
    passage_embeddings = match_model.encode(passage)

    similarity = match_model.similarity(query_embedding, passage_embeddings)

    sentence_similarity = similarity.tolist()[0]

    # create dict
    sentence_similarity = [(sentence.replace(" ", "_"), similarity) for sentence, similarity in zip(passage, sentence_similarity)]

    # Sort the list in descending order of similarity score
    #sentence_similarity.sort(key=lambda x: x[1], reverse=True)
    sentence_similarity_np = np.array(sentence_similarity)

    return sentence_similarity_np

def rank_labels_with_images(label, folder_path):

        prompt = f"Does all the 16 images contain {label}? y/n"

        # samples 16 random images from folder
        image_files = os.listdir(folder_path)
        image_files = [os.path.join(folder_path, img) for img in image_files]
        image_files = random.sample(image_files, 16)

        # open and create a 16 x 16 grid with the images
        images = [Image.open(img) for img in image_files]

        # Resize images to fit into a 4x4 grid
        width, height = images[0].size
        grid_size = (4, 4)
        grid_width = width * grid_size[0]
        grid_height = height * grid_size[1]
        grid_image = Image.new('RGB', (grid_width, grid_height))

        # Paste each image into the grid
        for index, image in enumerate(images):
            row = index // grid_size[0]
            col = index % grid_size[0]
            grid_image.paste(image, (col * width, row * height))

        # Example of creating a 4x4 grid as `raw_image`
        raw_image = grid_image

        inputs = gemma_processor(prompt, raw_image, return_tensors="pt").to(device)
        output = gemma_model.generate(**inputs, max_new_tokens=20)

        dec_out = gemma_processor.decode(output[0], skip_special_tokens=True)[len(prompt):]

        # remove /n from the output
        dec_out = dec_out.replace("\n", "")

        return dec_out, dec_out.lower() in ["y", "yes"]


def findBestMatch_many2many(queries, targets):

    #find correlations between queries and targets
    query_embeddings = match_model.encode(queries)
    passage_embeddings = match_model.encode(targets)

    similarity = match_model.similarity(query_embeddings, passage_embeddings).T

    #visualize the results on a confusion matrix
    fig, ax = plt.subplots()
    cax = ax.matshow(similarity, cmap='viridis')
    #label axis
    ax.set_xticks(range(len(queries)))
    ax.set_yticks(range(len(targets)))
    ax.set_xticklabels(queries, rotation=90)
    ax.set_yticklabels(targets)
    fig.colorbar(cax)
    plt.show()

def process_labels(im_labels, coco_class, hierarchy):
    count = len(im_labels)
    ret_labels = []
    queue = deque(im_labels)
    while queue:
        #print(f"children {len(ret_labels)}, going deeper...")
        #all_empty = True
        im_label = queue.popleft()
        lower_level_labels = hierarchy.get(im_label, [])
        if lower_level_labels:
            #all_empty = False
            ret_labels += lower_level_labels #add to the results
            count += len(lower_level_labels)

    return ret_labels

###################################################àà

#parse input
parser = argparse.ArgumentParser(description='Process some IMAGEMET.')
parser.add_argument('--data_id', type=str, default="selected2imagenet", help='data id')
parser.add_argument('--save_images', type=bool, default=False, help='save images')
args = parser.parse_args()

#### PROCESS LISTS

# Carica i file YAML
with open('./data_class_lists/selected_classes.yaml', 'r') as file:
    list1 = yaml.safe_load(file)

#reverse list1 dict
list1_reverse = {v: k for k, v in list1.items()}

with open('./data_class_lists/imagenet_cls.yaml', 'r') as file:
    list2 = yaml.safe_load(file)

## Estrai i nomi delle classi dalla lista 1
class_names_list1 = list(list1.values())


# Estrai i nomi delle classi dalla lista 2
class_names_list2 = list(list2.values())
class_names_list2 = [name.lower() for name in class_names_list2]

# Trova le corrispondenze tra le due liste
common_classes = set(class_names_list1).intersection(set(class_names_list2))

# Trova le classi che sono solo nella lista 1
only_in_list1 = set(class_names_list1) - common_classes

# Trova le classi che sono solo nella lista 2
only_in_list2 = set(class_names_list2) - common_classes

similarity_dict = {key : key for key in common_classes}

##############################################

#findBestMatch_many2many(class_names_list2[::100], class_names_list1)

class_names_list1_spaced = [sentence.replace("_", " ") for sentence in class_names_list1]
class_names_list2_spaced = [sentence.replace("_", " ") for sentence in class_names_list2]

#load imagenet hierarchy
hierarchy_file = "./data_class_lists/imagenet_cls_hierarchy.yaml"
with open(hierarchy_file, 'r') as file:
    hierarchy = yaml.safe_load(file)

res_dict = {coco_class : [] for coco_class in class_names_list1}

# check id file exists
data_id = args.data_id
save_images = args.save_images

out_file = f"mapping/{data_id}_out_dict.yaml"

if not os.path.exists(out_file):
    for coco_class in tqdm(class_names_list1):

        #search directly in the hierarchy
        im_labels = hierarchy.get(coco_class)

        # while im_labels and len(im_labels) < 10:
        #     print(f"<10 MATCHES FOR {coco_class}, going deeper...")
        #     all_empty = True
        #     for im_label in im_labels:
        #         lower_level_labels = hierarchy.get(im_label, [])
        #         if lower_level_labels:
        #             all_empty = False
        #         im_labels += lower_level_labels
        #     if all_empty:
        #         break

        if im_labels: #found a direct match
            res_dict[coco_class].append(coco_class)
            for im_label in im_labels:
                res_dict[coco_class].append(im_label)

            ret_labels = process_labels(im_labels, coco_class, hierarchy) #add children of first level children
            for im_label in ret_labels:
                res_dict[coco_class].append(im_label)


        else:
            #find the closest match
            coco_class_spaced = coco_class.replace("_", " ")
            im_label_spaced_list = findBestMatch(coco_class, class_names_list2_spaced)
            if im_label_spaced_list:
                im_labels = []
                for (im_label_spaced, score) in im_label_spaced_list:
                    im_label = im_label_spaced.replace(" ", "_")
                    im_labels.append(im_label)
                    res_dict[coco_class].append(im_label)
                    # lower_level_labels = hierarchy[im_label]
                    # if lower_level_labels:
                    #     for im_label in lower_level_labels:
                    #         res_dict[coco_class].append(im_label)
                ret_labels = process_labels(im_labels, coco_class, hierarchy)
                for im_label in ret_labels:
                    res_dict[coco_class].append(im_label)

            # else:
            #     print(f"NO MATCHES FOR {coco_class}")
            # else:
            #     print(f"NO MATCHES FOR {coco_class}")

            res_dict[coco_class] = list(set(res_dict[coco_class])) #remove replicas

    out_dict = {}
    #base path to folders
    #examples
    #statistics

    #base imagent path
    base_imagenet_path_train = '/media/data/Datasets/imagenet21k_resized/imagenet21k_train'
    base_imagenet_path_val = '/media/data/Datasets/imagenet21k_resized/imagenet21k_val'

    #open imagent mapping file
    with open('./data_class_lists/imagenet_cls.yaml', 'r') as file:
        mapping = yaml.safe_load(file)
        #invert the mapping
        mapping_inv = {v.lower(): k for k, v in mapping.items()}

    print("Processing results...")
    for key in tqdm(res_dict.keys()):
        #print(f"+++ COCO {key}: {len(res_dict[key])}")
        #print(res_dict[key])

        key_set = set(res_dict[key])

        folders = []
        folder_paths = []
        classes = []
        n_images_folder = 0
        examples = {}
        filtered_examples = {}
        scores = []

        #fill out_dict
        for im_label in key_set:
            if im_label in mapping_inv:
                folder = os.path.join(base_imagenet_path_train, mapping_inv[im_label])
                if os.path.exists(folder):
                    folder_paths.append(folder)
                    folders.append(folder)
                    classes.append(im_label)
                    n_images_folder += len(os.listdir(folder))
                    images_folder = folder
                    examples[im_label] = [os.path.join(folder,image) for image in random.sample(os.listdir(images_folder), 4)]
                else:
                    folder = folder.replace('train', 'val')
                    if not os.path.exists(folder):
                        print(f"Folder {folder} not found, skipping...")
                        continue
                    folder_paths.append(folder)
                    folders.append(folder)
                    classes.append(im_label)
                    n_images_folder += len(os.listdir(folder))
                    images_folder = folder.replace('train', 'val')
                    examples[im_label] = [os.path.join(folder,image) for image in random.sample(os.listdir(images_folder), 4)]
            #else:
            #    print(f"Folder for {im_label} not found...")

        #compute scores
        if classes:
            ranking = rank_label(key, classes)
            sorted_classes, scores = ranking[:,0].tolist(), ranking[:,1].astype(float).tolist()
        else:
            sorted_classes = []
            scores = []

        vis_checks = []
        raw_vis_checks = []
        for cls, im_fold in zip(classes, folder_paths):
            raw_out, res = rank_labels_with_images(key, im_fold)
            vis_checks.append(res)
            raw_vis_checks.append(raw_out)

        #filtered examples
        for im_label, fold in zip(np.array(classes)[vis_checks].tolist(), np.array(folders)[vis_checks].tolist()):
            filtered_examples[im_label] = [os.path.join(fold,image) for image in random.sample(os.listdir(fold), 4)]

        out_dict[key] = {
            "classes" : classes,
            "folders_paths" : folder_paths,
            "folders" : folders,
            "n_images" : n_images_folder,
            "n_classes" : len(folders),
            "examples" : examples,
            "scores" : scores,
            "vis_checks" : vis_checks,
            "raw_vis_checks" : raw_vis_checks,
            "filtered_examples" : filtered_examples
        }

    #save out_dict
    os.makedirs("mapping", exist_ok=True)
    with open(out_file, 'w') as file:
        yaml.dump(out_dict, file)
else:
    with open(out_file, 'r') as file:
        out_dict = yaml.safe_load(file)

#compute statistics on out_dict
empty_th = 10
statistics = {
    "empty": 0,
    "less_than_5_classes": 0,
    "less_then_TH_classes": 0,
    "more_then_TH_classes": 0,
    "total_im-classes_covered": 0,
    "total_im-images_covered": 0,
    "filtered_im-classes_covered": 0,
    "filtered_im-images_covered": 0,
    "filtered_more_then_TH_classes": 0,
    "filtered_more_than_5_classes": 0,
    "less_then_TH_class_list": {},
}
print("computing statistics...")
for key in tqdm(out_dict):

    vis_check = out_dict[key]["vis_checks"]

    if len(np.array(out_dict[key]["classes"])[vis_check]) >= empty_th:
        statistics["filtered_more_then_TH_classes"] += 1
        statistics["filtered_im-classes_covered"] += len(np.array(out_dict[key]["classes"])[vis_check])
        for fold in np.array(out_dict[key]["folders"])[vis_check]:
            statistics["filtered_im-images_covered"] += len(os.listdir(fold))

    if len(np.array(out_dict[key]["classes"])[vis_check]) > 5:
        statistics["filtered_more_than_5_classes"] += 1

    if len(out_dict[key]["classes"]) >= empty_th:
        statistics["more_then_TH_classes"] += 1
        statistics['total_im-classes_covered'] += len(out_dict[key]["classes"])
        statistics["total_im-images_covered"] += out_dict[key]["n_images"]
    else:
        statistics["less_then_TH_classes"] += 1
        statistics["less_then_TH_class_list"][key] = len(out_dict[key]["classes"])
        if len(out_dict[key]["classes"]) == 0:
            statistics["empty"] += 1
        if len(out_dict[key]["classes"]) < 5:
            statistics["less_than_5_classes"] += 1

#print statistics
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(statistics)

#save statistics
with open(f'mapping/{data_id}_statistics.yaml', 'w') as file:
    yaml.dump(statistics, file)

#check images
if save_images:
    print("saving images...")
    os.makedirs("mapping/imgs", exist_ok=True)
    for key in tqdm(out_dict):

        #print("Saving images for ", key)

        subclasses = out_dict[key]["classes"]
        rows = 4

        # Calculate the number of images to be saved
        num_images = math.ceil(len(subclasses) / 10)

        for img_num in range(num_images):
            # Calculate the start and end indices for slicing the subclasses list
            start_index = img_num * 10
            end_index = start_index + 10

            # Get the current chunk of subclasses
            current_subclasses = subclasses[start_index:end_index]
            cols = len(current_subclasses)

            plt.figure(f"{key}_{img_num}", figsize=(cols * 2, rows * 2))
            for j, subclass in enumerate(current_subclasses):
                images = out_dict[key]["examples"][subclass]
                for i, image in enumerate(images):
                    plt.subplot(rows, cols, i * (cols) + j + 1)
                    img = plt.imread(image)
                    plt.imshow(img)
                    pos_x, pos_y = 10, 10
                    if out_dict[key]["vis_checks"][j]:
                        plt.scatter(pos_x, pos_y, c="green", s=100, marker="o", edgecolors="white", linewidths=2)
                    else:
                        plt.scatter(pos_x, pos_y, c="red", s=100, marker="X", edgecolors="white", linewidths=2)
                    if i == 0:
                        plt.title(subclass)
                    plt.axis("off")

            plt.suptitle(key)
            plt.subplots_adjust(wspace=0.5)
            plt.tight_layout()
            try:
                plt.savefig(f"mapping/imgs/{key}_{img_num}.png")
            except:
                print(f"Error saving {key}_{img_num}")

            plt.close()





