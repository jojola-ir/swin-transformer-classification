import argparse
import os
from os.path import join

import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics

from dataloader import loader
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
        # Iterate over the data
        for batch_idx, (data, target) in enumerate(train_loader):
            # Move the data to the device
            data, target = data.to(device), target.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            output = model(data)
            pred = output.argmax(-1)

            # Compute the loss
            loss = criterion(output, target)

            # Backward pass
            loss.backward()

            # Update the weights
            optimizer.step()

            acc = accuracy(pred, target)
            rec = recall(pred, target)
            f1_score = f1(pred, target)

            # Print the loss
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\ttrain_accuracy: {:.6f}\ttrain_recall: {:.6f}\ttrain_f1: {:.6f}".format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item(),
                    acc, rec, f1_score), end="\r")

        if epoch % 5 == 0:
            torch.save(model.state_dict(), join(results_path, "model.pth"))

        validate(model, val_loader, criterion, num_classes, device)

    torch.save(model.state_dict(), join(results_path, "model.pth"))


def validate(model, val_loader, criterion, num_classes, device):
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

    # Iterate over the data
    for batch_idx, (data, target) in enumerate(val_loader):
        # Move the data to the device
        data, target = data.to(device), target.to(device)

        # Forward pass
        output = model(data)
        pred = output.argmax(-1)

        # Compute the loss
        loss = criterion(output, target)

        acc = accuracy(pred, target)
        rec = recall(pred, target)
        f1_score = f1(pred, target)

        # Print the error
        print(
            "Validation [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tval_accuracy: {:.6f}\tval_recall: {:.6f}\tval_f1: {:.6f}".format(
                batch_idx * len(data), len(val_loader.dataset),
                100. * batch_idx / len(val_loader), loss.item(), acc, rec, f1_score), end="\r")


def test(model, test_loader, num_classes, device):
    """Test the model.
    Args:
        model (nn.Module): The model.
        test_loader (DataLoader): The test loader.
        device (torch.device): The device to use.
    """
    # Set the model to evaluation
    model.eval()

    accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
    recall = torchmetrics.Recall(task="multiclass", num_classes=num_classes)
    f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)

    # Iterate over the data
    for batch_idx, (data, target) in enumerate(test_loader):
        # Move the data to the device
        data, target = data.to(device), target.to(device)

        # Forward pass
        output = model(data)
        pred = output.argmax(-1)

        acc = accuracy(pred, target)
        rec = recall(pred, target)
        f1_score = f1(pred, target)

        # Print the output
        print(
            "Test [{}/{} ({:.0f}%)]\tLoss: {:.6f}\ttest_accuracy: {:.6f}\ttest_recall: {:.6f}\ttest_f1: {:.6f}".format(
                batch_idx * len(data), len(test_loader.dataset),
                100. * batch_idx / len(test_loader), acc, rec, f1_score), end="\r")


def main(path_to_data, batch_size, epochs, lr, model_name, results_path, device):
    # Define the model
    model = build_model(model_name).to(device)

    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Define the data loaders
    train_loader, val_loader, test_loader, num_classes = loader(path_to_data, batch_size)
    print("Data loaded")

    # Train the model
    train(model, train_loader, val_loader, criterion, optimizer, epochs, num_classes, device, results_path)

    # Test the model
    test(model, test_loader, num_classes, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Swin Transformer Example")

    parser.add_argument("--path_to_data", type=str, default="./data", help="Path to the data")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--results", type=str, default="./results/", help="Path to results")
    parser.add_argument("--model_name", type=str, default="tiny", help="Model name")

    args = parser.parse_args()

    path_to_data = args.path_to_data
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    results_path = args.results
    model_name = args.model_name

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

    main(path_to_data, batch_size, epochs, lr, model_name, results_path, device)
