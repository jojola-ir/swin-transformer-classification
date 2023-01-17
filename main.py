import argparse
import os
from os.path import join

import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from tqdm import tqdm

from dataloader import split_loader
from metrics import AveragedHausdorffLoss
from model import build_model


def train(model, train_loader, val_loader, optimizer, epochs, num_classes, device, model_type, results_path=None):
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

    if model_type == "classification":
        criterion = nn.CrossEntropyLoss()
        accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        recall = torchmetrics.Recall(task="multiclass", num_classes=num_classes)
    elif model_type == "segmentation":
        criterion = nn.BCEWithLogitsLoss()
        hausdorff = AveragedHausdorffLoss()
    elif model_type == "regression":
        criterion = nn.MSELoss()
        mse = torchmetrics.MeanSquaredError()
    else:
        raise ValueError("Model type not recognized")

    f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)

    acc_list = []
    rec_list = []
    f1_list = []

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

                f1_score = f1(pred, target).numpy()

                # Print the loss and metrics
                if model_type == "classification":
                    acc = accuracy(pred, target).numpy()
                    rec = recall(pred, target).numpy()

                    acc_list.append(acc)
                    rec_list.append(rec)
                    f1_list.append(f1_score)

                    loader.set_postfix(loss=loss.item(), accuracy=acc_list / len(acc_list),
                                       recall=rec_list / len(rec_list),
                                       f1=f1_list / len(f1_list))
                elif model_type == "segmentation":
                    hausdorff_score = hausdorff(output, target).numpy()
                    loader.set_postfix(loss=loss.item(), f1=f1_score, hausdorff=hausdorff_score)
                elif model_type == "regression":
                    mse_score = mse(output, target).numpy()
                    loader.set_postfix(loss=loss.item(), mse=mse_score)

        if epoch % 1 == 0:
            torch.save(model.state_dict(), join(results_path, f"model_{model_type}.pth"))

        if epoch != epochs:
            validation(model, val_loader, num_classes, model_type, device)

    torch.save(model.state_dict(), join(results_path, f"model_{model_type}.pth"))


def validation(model, val_loader, num_classes, model_type, device):
    """Validate the model.
    Args:
        model (nn.Module): The model.
        val_loader (DataLoader): The validation loader.
        criterion (nn.Module): The loss function.
        device (torch.device): The device to use.
    """
    # Set the model to evaluation
    model.eval()

    if model_type == "classification":
        criterion = nn.CrossEntropyLoss()
        accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        recall = torchmetrics.Recall(task="multiclass", num_classes=num_classes)
    elif model_type == "segmentation":
        criterion = nn.BCEWithLogitsLoss()
        hausdorff = AveragedHausdorffLoss()
    elif model_type == "regression":
        criterion = nn.MSELoss()
        mse = torchmetrics.MeanSquaredError()
    else:
        raise ValueError("Model type not recognized")

    f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)

    acc_list = []
    rec_list = []
    f1_list = []

    with tqdm(val_loader, unit="batch") as loader:
        # Iterate over the data
        for data, target in loader:
            loader.set_description("Validation")
            # Move the data to the device
            data, target = data.to(device), target.to(device)

            # Forward pass
            output = model(data).logits
            pred = output.argmax(-1)
            f1_score = f1(pred, target).numpy()

            f1_list.append(f1_score)

            # Print the loss and metrics
            if model_type == "classification":
                acc = accuracy(pred, target).numpy()
                rec = recall(pred, target).numpy()

                acc_list.append(acc)
                rec_list.append(rec)
            elif model_type == "segmentation":
                hausdorff_score = hausdorff(output, target).numpy()
            elif model_type == "regression":
                mse_score = mse(output, target).numpy()

            if criterion is not None:
                # Compute the loss
                loss = criterion(output, target)

                # Print the loss and metrics
                if model_type == "classification":
                    loader.set_postfix(loss=loss.item(), accuracy=acc_list / len(acc_list),
                                       recall=rec_list / len(rec_list),
                                       f1=f1_list / len(f1_list))
                elif model_type == "segmentation":
                    loader.set_postfix(loss=loss.item(), f1=f1_score, hausdorff=hausdorff_score)
                elif model_type == "regression":
                    loader.set_postfix(loss=loss.item(), mse=mse_score)
            else:
                if model_type == "classification":
                    loader.set_postfix(accuracy=acc_list / len(acc_list),
                                       recall=rec_list / len(rec_list),
                                       f1=f1_list / len(f1_list))
                elif model_type == "segmentation":
                    loader.set_postfix(f1=f1_score, hausdorff=hausdorff_score)
                elif model_type == "regression":
                    loader.set_postfix(mse=mse_score)


def main(path_to_data, batch_size, epochs, lr, model_name, img_size, results_path, model_type, device):
    # Define the model
    model = build_model(model_name=model_name, model_type=model_type).to(device)

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Define the data loaders
    train_loader, val_loader, test_loader, num_classes = split_loader(path_to_data, img_size, batch_size)
    print("Data loaded")

    print(model)

    p = sum([p.numel() for p in model.parameters()])
    print(f"Number of parameters: {p}")

    # # Train the model
    # print("Training started")
    # train(model, train_loader, val_loader, optimizer, epochs, num_classes, device, results_path)
    #
    # # Test the model
    # validation(model, val_loader, num_classes, model_type, device)


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
    parser.add_argument("--regression", "-r", action="store_true", help="Regression")

    args = parser.parse_args()

    path_to_data = args.path_to_data
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    results_path = args.results
    img_size = args.img_size
    model_name = args.model_name
    segmentation = args.segmentation
    regression = args.regression

    if segmentation:
        model_type = "segmentation"
    elif regression:
        model_type = "regression"
    else:
        model_type = "classification"

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

    main(path_to_data, batch_size, epochs, lr, model_name, img_size, results_path, model_type, device)
