#!/bin/bash

# Check if all required parameters are provided
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <data_id> <root_imagenet> <batch_size>"
    exit 1
fi

# Assign command-line arguments to variables
DATA_ID=$1
ROOT_IMAGENET=$2
BATCH_SIZE=$3

# Classification Dataset Creation

# Step 1: Cluster ImageNet using WordNet and Sentence Transformer
echo "Running cluster_imagenet.py..."
python cluster_imagenet.py

# Step 2: Build VLM Superclass by associating sub-classes to the proper super-class
echo "Running vlm_superclass_building.py... [HUMAN INTERACTION REQUIRED]"
python vlm_superclass_building.py --data_id "$DATA_ID" --root_imagenet "$ROOT_IMAGENET"

# Step 3: Perform human checks on the images
echo "Running human_check_tool.py... [HUMAN INTERACTION REQUIRED]"
python human_check_tool.py --data_id "$DATA_ID"

# Step 4: Check for replicated sub-classes in each super-class
echo "Running check_replicas.py... [HUMAN INTERACTION REQUIRED]"
python check_replicas.py --data_id "$DATA_ID"

# Step 5: Filter super-classes with few sub-classes
echo "Running check_scores.py..."
python check_scores.py --data_id "$DATA_ID" --min_num_subclasses 10

# Step 6: Compute correlation scores with CLIP
echo "Running clip_score_generation.py..."
python clip_score_generation.py --data_id "$DATA_ID" --root_imagenet "$ROOT_IMAGENET" --batch_size "$BATCH_SIZE"

# Step 7: Detect outliers in the score distribution
echo "Running outliers_detection.py..."
python outliers_detection.py

# Step 8: Split the dataset into IID (train), test easy OOD, and test hard OOD
echo "Running dataset_split.py..."
python dataset_split.py --root_imagenet "$ROOT_IMAGENET" --p_value_1 40 --p_value_2 20

echo "All steps completed."
