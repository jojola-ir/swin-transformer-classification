import os
from os.path import join

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split


def get_noisy_image(image, noise_parameter=0.2):
    """Adds noise to an image.

    Args:
        image: image, np.array with values from 0 to 1
    """
    image_shape = image.shape

    # noise_type = np.random.choice(['gaussian', 'poisson', 'bernoulli'])
    # if noise_type == 'gaussian':
    #     noise = torch.normal(0, noise_parameter, image_shape)
    #     noisy_image = (image + noise).clip(0, 1)
    # elif noise_type == 'poisson':
    #     a = noise_parameter * torch.ones(image_shape)
    #     noise = torch.poisson(a)
    #     noise /= noise.max()
    #     noisy_image = (image + noise).clip(0, 1)
    # elif noise_type == 'bernoulli':
    #     noise = torch.bernoulli(noise_parameter * torch.ones(image_shape))
    #     noisy_image = (image * noise).clip(0, 1)

    noise = torch.normal(0, noise_parameter, image_shape)
    noisy_image = (image + noise).clip(0, 1)

    return noisy_image


def transformations(img_size):
    """Applies transformations to an image.

    Args:
        image: image, np.array
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.ConvertImageDtype(torch.float),
        transforms.Resize((img_size, img_size)),
        transforms.Normalize(mean=(0,),
                             std=(1,)),
    ])
    return transform


def split_loader(path, img_size, batch_size):
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
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

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
