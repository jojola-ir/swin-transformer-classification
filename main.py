import argparse
import os
from os.path import join

import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from tqdm import tqdm

from dataloader import split_loader
from model import build_model


def train(model, train_loader, val_loader, criterion, optimizer, epochs, num_classes, device, results_path=None):
    """Train the model.
    Args:
        model (nn.Module): The model.
        train_loader (DataLoader): The train loader.
        criterion (nn.Module): The loss function.
        optimizer (nn.Module): The optimizer.
        device (torch.device): The device to use.
    """
    # Set the model to training mode
    model.train()

    accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
    recall = torchmetrics.Recall(task="multiclass", num_classes=num_classes)
    f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)

    for epoch in range(1, epochs + 1):
        with tqdm(train_loader, unit="batch") as loader:
        # Iterate over the data
            for data, target in loader:
                loader.set_description(f"Epoch {epoch}")
                # Move the data to the device
                data, target = data.to(device), target.to(device)

                # Zero the gradients
                optimizer.zero_grad()

                # Forward pass
                output = model(data).logits
                pred = output.argmax(-1)

                # Compute the loss
                loss = criterion(output, target)

                # Backward pass
                loss.backward()

                # Update the weights
                optimizer.step()

                acc = accuracy(pred, target).numpy()
                rec = recall(pred, target).numpy()
                f1_score = f1(pred, target).numpy()

                # Print the loss and metrics
                loader.set_postfix(loss=loss.item(), accuracy=acc, recall=rec, f1=f1_score)

        if epoch % 5 == 0:
            torch.save(model.state_dict(), join(results_path, "model.pth"))

        validation(model, val_loader, criterion, num_classes, device)

    torch.save(model.state_dict(), join(results_path, "model.pth"))


def validation(model, val_loader, criterion, num_classes, device):
    """Validate the model.
    Args:
        model (nn.Module): The model.
        val_loader (DataLoader): The validation loader.
        criterion (nn.Module): The loss function.
        device (torch.device): The device to use.
    """
    # Set the model to evaluation
    model.eval()

    accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
    recall = torchmetrics.Recall(task="multiclass", num_classes=num_classes)
    f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)

    with tqdm(val_loader, unit="batch") as loader:
        # Iterate over the data
        for data, target in loader:
            loader.set_description("Validation")
            # Move the data to the device
            data, target = data.to(device), target.to(device)

            # Forward pass
            output = model(data).logits
            pred = output.argmax(-1)

            acc = accuracy(pred, target).numpy()
            rec = recall(pred, target).numpy()
            f1_score = f1(pred, target).numpy()

            if criterion is not None:
                # Compute the loss
                loss = criterion(output, target)

                # Print the loss and metrics
                loader.set_postfix(loss=loss.item(), accuracy=acc, recall=rec, f1=f1_score)
            else:
                loader.set_postfix(accuracy=acc, recall=rec, f1=f1_score)

def main(path_to_data, batch_size, epochs, lr, model_name, img_size, results_path, segmentation, device):
    # Define the model
    model = build_model(model_name=model_name, segmentation=segmentation).to(device)

    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Define the data loaders
    train_loader, val_loader, test_loader, num_classes = split_loader(path_to_data, img_size, batch_size)
    print("Data loaded")

    if not segmentation:
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    print(model)

    p = sum([p.numel() for p in model.parameters()])
    print(f"Number of parameters: {p}")

    print(f"Number of classes: {num_classes}")

    # # Train the model
    # print("Training started")
    # train(model, train_loader, val_loader, criterion, optimizer, epochs, num_classes, device, results_path)
    #
    # # Test the model
    # validation(model, val_loader, None, num_classes, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Swin Transformer reimplementation")

    parser.add_argument("--path_to_data", type=str, default="./data", help="Path to the data")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--results", type=str, default="./results/", help="Path to results")
    parser.add_argument("--img_size", type=int, default=224, help="Image size")
    parser.add_argument("--model_name", type=str, default="tiny", help="Model name")
    parser.add_argument("--segmentation", "-s", action="store_true", help="Segmentation")

    args = parser.parse_args()

    path_to_data = args.path_to_data
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    results_path = args.results
    img_size = args.img_size
    model_name = args.model_name
    segmentation = args.segmentation

    # Create the results directory
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # Use the GPU if available
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    main(path_to_data, batch_size, epochs, lr, model_name, img_size, results_path, segmentation, device)
