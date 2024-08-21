import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as transforms
from torchvision import tv_tensors as tv
import matplotlib.pyplot as plt
import os
from utils.custom_transforms import InnerRandomCrop, Resize
from torch.utils.data import DataLoader


class SOODImageNetC(Dataset):
    def __init__(self, file_list, base_path=None, transform=None, mode="train_val", resize=(256, 256), augmenter=None):
        """
        Args:
            file_list (list): list of image files with labels in format ['relative_path/to/image label class_name subclass_name']
            base_path (str): path to the folder where images are stored
            transform: custom transform, if None default transforms will be used (Crop, Resize, Normalize)
            mode (str): 'train_val' or 'test'
            resize (tuple): final size of images with defualt transform, default=(256, 256)
            dataset_id (str): dataset id
            augmenter: add an augmentation policy from torchvision.transforms.v2, such as AugMix
        """

        self.file_list = file_list
        if transform is None:
            self.use_default_transform = True
        else:
            self.use_default_transform = False

        self.base_path = base_path
        self.augmenter = augmenter
        self.transform = transform

        #self.label2index = label2index

        assert mode in ["train_val", "test"], "Mode should be either 'train_val' or 'test'"
        self.mode = mode
        self.resize = resize

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path, label, class_name, subclass_name = self.file_list[idx].split()

        if self.base_path:
            img_path = os.path.join(self.base_path, img_path)

        # read image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Image not found: {img_path}")
            return None, None, None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = torch.as_tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0

        if not self.use_default_transform:
            t_img = self.transform(img)
        else:
            smaller_dim = min(img.shape[1:])
            if self.mode == "train_val":
                if self.augmenter:
                    self.transform = transforms.Compose(
                        [
                            InnerRandomCrop(smaller_dim, smaller_dim),
                            Resize(self.resize[0], self.resize[1]),
                            self.augmenter,
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ]
                    )
                else:
                    self.transform = transforms.Compose(
                        [
                            InnerRandomCrop(smaller_dim, smaller_dim),
                            Resize(self.resize[0], self.resize[1]),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ]
                    )
            else:
                self.transform = transforms.Compose(
                    [
                        transforms.CenterCrop(smaller_dim),
                        Resize(self.resize[0], self.resize[1]),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ]
                )

            t_img = self.transform(img)

            #debug
            # plt.subplot(1, 2, 1)
            # plt.imshow(img.permute(1, 2, 0))
            # plt.title("Original Image")
            # plt.axis("off")
            # plt.subplot(1, 2, 2)
            # #denorn t_img
            # t_img_d = t_img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            # plt.imshow(t_img_d.permute(1, 2, 0))
            # plt.title("Transformed Image")
            # plt.axis("off")
            # plt.show()

        ret_label = int(label)

        return t_img, ret_label, class_name, subclass_name

class SOODImageNetS(Dataset):
    def __init__(self, file_list, base_path=None, mask_base_path=None, transform=None, max_classes=57, mode="train_val", resize=(256, 256)):

        self.file_list = file_list
        if transform is None:
            self.use_default_transform = True
        else:
            self.use_default_transform = False

        self.base_path = base_path
        self.max_classes = max_classes - 1 #remove background class
        self.transform = transform

        if mask_base_path is None:
            self.mask_base_path = base_path
        else:
            self.mask_base_path = mask_base_path

        assert mode in ["train_val", "test"], "Mode should be either 'train_val' or 'test'"
        self.mode = mode
        self.resize = resize

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        data = self.file_list[idx].split()

        # Read image and mask
        img_path, mask_path, label, class_name, subclass_name = data

        if self.base_path:
            img_path = os.path.join(self.base_path, img_path)
            mask_path = os.path.join(self.mask_base_path, mask_path)

        # read image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Image not found: {img_path}")
            return None, None, None, None, None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Read mask
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

        if mask is None:
            print(f"Mask not found: {mask_path}")
            return None, None, None, None, None

        mask = tv.Mask(mask)

        if mask.max() > self.max_classes:
            print(f"Mask has more than {self.max_classes} classes")
            return None, None, None, None, None

        img = torch.as_tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0

        if not self.use_default_transform:
            t_img, t_mask = self.transform(img, mask)
        else:
            smaller_dim = min(img.shape[1:])
            if self.mode == "train_val":
                self.transform = transforms.Compose(
                    [
                        InnerRandomCrop(smaller_dim, smaller_dim),
                        Resize(self.resize[0], self.resize[1]),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ]
                )
            else:
                self.transform = transforms.Compose(
                    [
                        transforms.CenterCrop(smaller_dim),
                        Resize(self.resize[0], self.resize[1]),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ]
                )

            try:
                t_img, t_mask = self.transform(img, mask)
            except Exception as e:
                print(f"Error transforming image: {e}")
                print(f"Image path: {img_path}")
                print(f"Mask path: {mask_path}")
                return None, None, None, None, None

        #visualize for debug
        # plot([[img, t_img],[mask, t_mask]], row_title=["Image", "Mask"])
        # plt.show()

        return t_img, t_mask, label, class_name, subclass_name


# HELPER STATIC METHODS
def get_loaders(task, mode, file_list_path, base_path, batch_size, workers, mask_base_path=None, input_shape=(224,244), transform=None, augmenter=None):
    """
    Args:
        task (str): 'classification' or 'segmentation'
        mode (str): 'train_val' or 'test'
        file_list_path (str): path to the file list with image paths and labels
        base_path (str): path to the folder where images are stored
        dataset_id (str): dataset id
        batch_size (int): batch size
        workers (int): number of workers
        input_shape (tuple): final size of images with defualt transform, default=(224, 224)
        transform: custom transform, if None default transforms will be used (Crop, Resize, Normalize)
        augmenter: add an augmentation policy from torchvision.transforms.v2, such as AugMix (for mode='train_val' only)
    Returns:
        train_loader, val_loader (for mode='train_val') or test_loader (for mode='test')
    """

    assert mode in ["train_val", "test"], "Mode should be either 'train_val' or 'test'"
    assert task in ["classification", "segmentation"], "Task should be either 'classification' or 'segmentation'"

    if mode == "train_val":
        # Read file list from txt file
        with open(file_list_path, 'r') as f:
            file_list = f.read().splitlines()

        # Create dataset
        if task == "classification":
            dataset = SOODImageNetC(file_list,
                                     base_path=base_path,
                                     transform=transform,
                                     mode=mode,
                                     resize=input_shape,
                                     augmenter=augmenter)
        else:
            dataset = SOODImageNetS(file_list,
                                     base_path=base_path,
                                     transform=transform,
                                     mode=mode,
                                     resize=input_shape,
                                     mask_base_path=mask_base_path)


        # Split dataset into train and validation sets
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        # Define data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=workers)

        return train_loader, val_loader

    else:
        with open(file_list_path, 'r') as f:
            file_list = f.read().splitlines()

        if task == "classification":
            test_dataset = SOODImageNetC(file_list,
                                              base_path=base_path,
                                              transform=transform,
                                              mode=mode,
                                              resize=input_shape)
        else:
            test_dataset = SOODImageNetS(file_list,
                                              base_path=base_path,
                                              transform=transform,
                                              mode=mode,
                                              resize=input_shape,
                                              mask_base_path=mask_base_path)

        # Create data loader
        test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=workers, shuffle=False)

        return test_loader


