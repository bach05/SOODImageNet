import matplotlib.pyplot as plt
from utils.SOODImageNetDataset import get_loaders
import random
import torchvision.transforms.v2 as transforms
from utils.custom_transforms import InnerRandomCrop, Resize
import argparse
import os
def vis_batch(loader, mask=False):
    # Get one batch of training images
    if mask:
        images, masks, labels, class_names, subclass_names = next(iter(loader))
    else:
        images, labels, class_names, subclass_names = next(iter(loader))

    # visualize the batch
    fig = plt.figure()
    for idx in range(batch_size):

        if mask:
            img = images[idx].permute(1, 2, 0)
            seg_mask = masks[idx]

            # Show original image
            ax = fig.add_subplot(2, batch_size, idx + 1, xticks=[], yticks=[])
            ax.imshow(img)
            ax.set_title(f"superclass: {class_names[idx]} - subclass: {subclass_names[idx]}")

            # Show masked image below the original image
            ax = fig.add_subplot(2, batch_size, batch_size + idx + 1, xticks=[], yticks=[])
            ax.imshow(img)
            ax.imshow(seg_mask, alpha=0.5)
        else:
            img = images[idx].permute(1, 2, 0)
            ax = fig.add_subplot(1, batch_size, idx + 1, xticks=[], yticks=[])
            ax.imshow(img)
            ax.set_title(f"superclass: {class_names[idx]} - subclass: {subclass_names[idx]}")

    plt.show()

if __name__ == "__main__":

    #Arguments
    parser = argparse.ArgumentParser(description='Test the loaders')
    parser.add_argument('--batch_size', type=int, default=2,    help='Batch size')
    parser.add_argument('--workers', type=int, default=2,    help='Number of workers')
    parser.add_argument('--cls_list_path', type=str, default="lists/classification",    help='Path to the classification list')
    parser.add_argument('--seg_list_path', type=str, default="lists/segmentation",    help='Path to the segmentation list')
    parser.add_argument('--iid_list', type=str, default="train_iid.txt",    help='Path to the iid list')
    parser.add_argument('--ood_easy_list', type=str, default="test_easy_ood.txt",    help='Path to the easy ood list')
    parser.add_argument('--ood_hard_list', type=str, default="test_hard_ood.txt",    help='Path to the hard ood list')
    parser.add_argument('--base_path', type=str, default="/media/data/Datasets/imagenet21k_resized/",    help='Path to the dataset')
    parser.add_argument('--mask_base_path', type=str, default="/media/data/Datasets/imagenet21k_resized/",    help='Path to the masks')

    args = parser.parse_args()

    path_iid_list = os.path.join(args.cls_list_path, args.iid_list)
    path_ood_easy_list = os.path.join(args.cls_list_path, args.ood_easy_list)
    path_ood_hard_list = os.path.join(args.cls_list_path, args.ood_hard_list)
    base_path = args.base_path

    batch_size = args.batch_size
    workers = args.workers
    input_shape = (224, 224)

    transform = transforms.Compose(
                            [
                                Resize(input_shape[0], input_shape[1]),
                            ])

    train_loader, val_loader = get_loaders('classification', "train_val",
                                    path_iid_list, base_path, batch_size, workers, input_shape=input_shape)
    test_loader_easy = get_loaders('classification', "test",
                                   path_ood_easy_list, base_path, batch_size, workers, transform=transform)
    test_loader_hard = get_loaders('classification', "test",
                                   path_ood_hard_list, base_path, batch_size, workers, transform=transform)

    # Visualize some images
    print("+++++ CLASSIFICATION +++++")
    print("Training images test....")
    vis_batch(train_loader)
    print("Validation images test....")
    vis_batch(val_loader)
    print("Test easy images test....")
    vis_batch(test_loader_easy)
    print("Test hard images test....")
    vis_batch(test_loader_hard)

    # Segmentation
    path_iid_list = os.path.join(args.seg_list_path, args.iid_list)
    path_ood_easy_list = os.path.join(args.seg_list_path, args.ood_easy_list)
    path_ood_hard_list = os.path.join(args.seg_list_path, args.ood_hard_list)
    base_path = args.base_path
    mask_base_path = args.mask_base_path

    train_loader, val_loader = get_loaders('segmentation', "train_val",
                                    path_iid_list, base_path, batch_size, workers, mask_base_path, input_shape=input_shape)
    test_loader_easy = get_loaders('segmentation', "test",
                                    path_ood_easy_list, base_path, batch_size, workers, mask_base_path, transform=transform)
    test_loader_hard = get_loaders('segmentation', "test",
                                    path_ood_hard_list, base_path, batch_size, workers, mask_base_path, transform=transform)

    # Visualize some images
    print("+++++ SEGMENTATION +++++")
    print("Training images test....")
    vis_batch(train_loader, mask=True)
    print("Validation images test....")
    vis_batch(val_loader, mask=True)
    print("Test easy images test....")
    vis_batch(test_loader_easy, mask=True)
    print("Test hard images test....")
    vis_batch(test_loader_hard, mask=True)

    print("All tests passed!")

