import glob
import os
from os.path import join

import imageio.v2 as imageio
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, random_split, DataLoader


def get_noisy_image(image, noise_parameter=0.2):
    """Adds noise to an image.

    Args:
        image: image, np.array with values from 0 to 1
    """
    image_shape = image.shape

    noise = torch.normal(0, noise_parameter, image_shape)
    noisy_image = (image + noise).clip(0, 1)

    return noisy_image


def train_transformation(img_size):
    """Applies transformations to an image.

    Args:
        image: image, np.array
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225]),
    ])
    return transform


def test_transformation(img_size):
    """Applies transformations to an image.

    Args:
        image: image, np.array
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225]),
    ])
    return transform


def split_classification_loader(path, img_size, batch_size):
    """Load the dataset.
    Args:
        path (str): The path to the dataset.
        batch_size (int): The batch size.
    Returns:
        train_loader (DataLoader): The train loader.
        val_loader (DataLoader): The validation loader.
        test_loader (DataLoader): The test loader.
    """

    # Define the transforms
    transform_train = train_transformation(img_size)
    transform_test = test_transformation(img_size)

    # Load the dataset
    train_path = join(path, "train/")
    test_path = join(path, "test/")

    train_dataset = torchvision.datasets.ImageFolder(
        root=train_path, transform=transform_train)
    test_dataset = torchvision.datasets.ImageFolder(
        root=test_path, transform=transform_test)

    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size

    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    totalDir = 0
    for base, dirs, files in os.walk(train_path):
        print('Searching in : ', base)
        for directories in dirs:
            if directories != ".DS_Store":
                totalDir += 1

    num_classes = totalDir if totalDir != 0 else 2

    # Create the data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, num_classes


class CustomDataLoader(Dataset):
    def __init__(self, folder_path, img_size, dataset_type, model_type):
        super(CustomDataLoader, self).__init__()
        self.img_files = glob.glob(os.path.join(folder_path, "images/", "*.png"))
        self.mask_files = []

        for img_path in self.img_files:
            if model_type == "classification" or model_type == "segmentation":
                self.mask_files.append(os.path.join(folder_path, "masks/", os.path.basename(img_path)))
            elif model_type == "regression":
                self.mask_files = self.img_files

        self.transform_train = train_transformation(img_size)
        self.transform_test = test_transformation(img_size)

        self.dataset_type = dataset_type

    def __getitem__(self, index):
        img_path = self.img_files[index]
        mask_path = self.mask_files[index]
        if self.dataset_type == "train":
            data = self.transform_train(Image.open(img_path).convert("L"))
            label = self.transform_train(Image.open(mask_path).convert("L"))
        elif self.dataset_type == "val" or self.dataset_type == "test":
            data = self.transform_test(Image.open(img_path).convert("L"))
            label = self.transform_test(Image.open(mask_path).convert("L"))
        else:
            raise ValueError("Invalid dataset type")

        data /= 255.0
        label /= 255.0

        return data, label

    def __len__(self):
        return len(self.img_files)


def get_noisy_image(image, noise_parameter=0.2):
    """Adds noise to an image.

    Args:
        image: image, np.array with values from 0 to 1
    """
    image_shape = image.shape

    noise = torch.normal(0, noise_parameter, image_shape)
    noisy_image = (image + noise).clip(0, 1)

    return noisy_image


if __name__ == "__main__":
    path = "/Users/irina/Documents/Etudes/DS/datasets/GAN/monet_style_dataset/All/"

    img_size = 224
    batch_size = 8
    img = imageio.imread(join(path, "train/images/image37.png"), pilmode='RGB')
    # print(img.shape)

    train_dataset = CustomDataLoader(join(path, "train/"), img_size, dataset_type="train")
    val_dataset = CustomDataLoader(join(path, "val/"), img_size, dataset_type="val")
    test_dataset = CustomDataLoader(join(path, "test/"), img_size, dataset_type="test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
    )

    for data, target in train_loader:
        print(f"Data shape: {data.shape}")
        break
