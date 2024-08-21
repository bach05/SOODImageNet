# SOOD-ImageNet: a Large-Scale Dataset for Semantic Out-Of-Distribution Image Classification and Semantic Segmentation

**ABSTRACT:**

Out-of-Distribution (OOD) detection in computer vision is a crucial research area, with related benchmarks playing a vital role in assessing the generalizability of models and their applicability in real-world scenarios. However, existing OOD benchmarks in the literature suffer from two main limitations: (1) they often overlook semantic shift as a potential challenge, and (2) their scale is limited compared to the large datasets used to train modern models. To address these gaps, we introduce SOOD-ImageNet, a novel dataset comprising around 1.6M images across 56 classes, designed for common computer vision tasks such as image classification and semantic segmentation under OOD conditions, with a particular focus on the issue of semantic shift. We ensured the necessary scalability and quality by developing an innovative data engine that leverages the capabilities of modern vision-language models, complemented by accurate human checks.

![cover_image](media/train_test_examples.png)

## Citation

If you use this dataset in your research, please cite the following paper:

``` bibtex
@article{SOOD-ImageNet,
  title={SOOD-ImageNet: a Large-Scale Dataset for Semantic Out-Of-Distribution Image Classification and Semantic Segmentation},
  }
```

## Installation

```commandline
git clone
cd SOOD-ImageNet-dataset
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 
pip install -r requirements.txt
```
## Download and Usage of pre-compiled lists 

We provide the lists used in the paper "*SOOD-ImageNet: a Large-Scale Dataset for Semantic Out-Of-Distribution Image Classification and Semantic Segmentation*" for our experiments. 

### Download ImageNet-21K-P

The images can be obtained from the official website: [Download ImageNet](http://www.image-net.org/). 

We used IMAGENET-21K-P dataset. After the download, your file system should look like: 

```
\download_root
   └── imagenet21k_resized
       ├── imagenet21k_train
       │   ├── n00005787
       │   ├── n02105162
       │   └── ... 
       └── imagenet21k_val
           └── ...
```

### Download of SOOD-ImageNet lists

The lists are available in the `lists.zip` file of this repository. Just run:
```commandline
unzip lists.zip
```
#### Additiional information

Note that the lists are provided as `.txt` files. The `lists.zip` compressed folder contains 2 folders. 

***For the image classification task***. The folder `classification` contains the following files: 
- `classification/train_iid.txt`, with the images we used for the *IID training*
- `classification/test_easy_ood.txt`, with the images we used for *OOD test* with a smaller semantic shift
- `classification/test_hard_ood.txt`, with the images we used for *OOD test* with a larger semantic shift

Each line of the list is structured as follows: 
```
imagenet21k_train/[synset_folder]/[image_file].JPG [class_ID] [superclass_name] [subclass_name]
```

***For the semantic segmentation task***. The folder `segmentation` contains the following files: 
- `segmentation/train_iid.txt`, with the images we used for the *IID training*
- `segmentation/test_easy_ood.txt`, with the images we used for *OOD test* with a smaller semantic shift
- `segmentation/test_hard_ood.txt`, with the images we used for *OOD test* with a larger semantic shift

Each line of the list is structured as follows: 
```
imagenet21k_train/[synset_folder]/[image_file].JPG output_test_easy_sam2/[synset_folder]/[image_file]_mask.png [class_ID] [superclass_name] [subclass_name]
```
Note that the segmentation masks are given as PNG files of shape `HxW` where each pixel is an integer representing the class ID (0 for background).

**DOWNLOAD PRE-LABELLED MASKS**: [TRAIN IID](https://drive.google.com/file/d/13o1dMAa56TqOTHyOf4gf6dFPheARqpoC/view?usp=sharing) | [TEST EASY OOD](https://drive.google.com/file/d/1AppoFP8EsPMv3pjwkmKH8ENm_FAys7UG/view?usp=drive_link) | [TEST HARD OOD](https://drive.google.com/file/d/1RqJSUjdWniBDG3dXmaF_PTbaEDQE2FEk/view?usp=drive_link)

## Usage

We provide torch datasets to load the images and labels for the image classification and semantic segmentation tasks in the `utils` folder.

If everything is set up correctly, you can run the following code to test the datasets:
```commandline
python check_loader.py --base_path download_root
```

You easily import the datasets in your project. For example, to load the image classification dataset:
```python
from utils.SOODImageNet import get_loaders, SOODImageNetC
```

## Data Engine 

If you would like to use the data engine to create your own lists from scratch, you can follow the instructions below.

### Before using

Download PaliGemma model from [Hugging Face](https://huggingface.co/google/paligemma-3b-mix-224) and save it in the `hf_models` folder:
```commandline
git lfs install
mkdir hf_models
cd hf_models
git clone https://huggingface.co/google/paligemma-3b-mix-224
```

### Usage

*NOTE: tested on RTX 4090 24GB, Pytorch 2.3.1, CUDA 11.8, Python 3.10.12*

**TO DO**:
- remove useless files
- check and remove commented code
- remove hard coded paths
- add parameters guide for each script

Needed files:
- `data_class_lists/imagenet_cls.yaml`, contains class names for each synset
- `cluster_images.py`, contains the code to create the hierarchical structure of ImageNet-21K-P using WordNet and Sentence Transformer

- `data_class_lists/selected_classes.yaml`, contains the selected super-classes for the SOOD-ImageNet dataset (note that you can define your own classes if you like, but it is not guaranteed to have all of them in the final dataset due to the filtering process)
- `vlm_superclass_building.py`, contains the code to create associate the sub-classes to the proper super-class (**TO DO**: remove hard coded paths (line 279))

- `human_check_tool.py`, contains the code to perform the human checks on the images. You can interrupt the labelling and resume it. 

- `check_replicas.py`, contains the code to check for replicated sub-classes each super-class. The user is asked to select in which super-class to keep the sub-class.

- `check_scores.py`, contains the code to filter the super-classes with a few sub-classes. Add also old scoring system. **TO DO**: add scoring with CLIP

- Final out file `mapping/{data_id}_sub_{min_num_subclasses}_split_{split_sizes[0]}-{split_sizes[1]}.yaml`

- Extract lists... 

## Classification Experiments

## Segementation Experiments



