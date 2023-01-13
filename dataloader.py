from os.path import join

import torch
import torchvision
import torchvision.transforms as transforms


def loader(path, batch_size):
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
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Load the dataset
    train_dataset = torchvision.datasets.ImageFolder(
        root=join(path, "train/"), transform=transform_train)
    val_dataset = torchvision.datasets.ImageFolder(
        root=join(path, "valid/"), transform=transform_test)
    test_dataset = torchvision.datasets.ImageFolder(
        root=join(path, "test/"), transform=transform_test)

    num_classes = len(train_dataset.classes)

    # Create the data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, num_classes
