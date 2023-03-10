import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from os.path import join

import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from torch.utils.data import DataLoader
from torchmetrics import PeakSignalNoiseRatio as PSNR
from tqdm import tqdm

from dataloader import CustomDataLoader, get_noisy_image, split_classification_loader
from losses import DiceLoss
from metrics import DiceScore
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
        criterion = DiceLoss()
    elif model_type == "regression":
        criterion = nn.MSELoss()
        mse = torchmetrics.MeanSquaredError()
        psnr = PSNR()
    else:
        raise ValueError("Model type not recognized")

    f1 = DiceScore()

    loss_list = []

    acc_list = []
    rec_list = []
    f1_list = []

    for epoch in range(1, epochs + 1):
        with tqdm(train_loader, unit="batch") as loader:
            # Iterate over the data
            for data, target in loader:
                loader.set_description(f"Epoch {epoch} / {epochs}")
                # Move the data to the device
                if model_type == "classification" or model_type == "segmentation":
                    data, target = data.to(device), target.to(device)
                elif model_type == "regression":
                    data, target = data.to(device), get_noisy_image(target).to(device)

                # Zero the gradients
                optimizer.zero_grad()

                # Forward pass
                if model_type == "classification":
                    output = model(data).logits
                    pred = output.argmax(-1)
                else:
                    output = model(data)
                    pred = output

                # Compute the loss
                loss = criterion(output, target)
                loss_list.append(loss.item())

                # Backward pass
                loss.backward()

                # Update the weights
                optimizer.step()

                f1_score = f1(pred.detach(), target.detach()).numpy()
                f1_list.append(f1_score)

                # Print the loss and metrics
                if model_type == "classification":
                    acc = accuracy(pred.detach(), target.detach()).numpy()
                    rec = recall(pred.detach(), target.detach()).numpy()

                    acc_list.append(acc)
                    rec_list.append(rec)
                    f1_list.append(f1_score)

                    loader.set_postfix(loss=loss.item(), accuracy=np.array(acc_list) / len(acc_list),
                                       recall=np.array(rec_list) / len(rec_list),
                                       f1=np.array(f1_list) / len(f1_list))
                elif model_type == "segmentation":
                    loader.set_postfix(loss=loss.item(), f1=f1_score, data_min=data.min().item(),
                                       data_max=data.max().item())
                elif model_type == "regression":
                    mse_score = mse(output.detach(), target.detach()).numpy()
                    train_psnr = psnr(pred.detach(), target.detach()).numpy()
                    loader.set_postfix(loss=loss.item(), mse=mse_score, psnr=train_psnr)

        if epoch % 1 == 0:
            torch.save(model.state_dict(), join(results_path, f"model_{model_type}.pth"))

        if epoch != epochs:
            validation(model, val_loader, num_classes, model_type, device)

    torch.save(model.state_dict(), join(results_path, f"model_{model_type}.pth"))

    fig, ax = plt.subplots()
    ax.plot(loss_list, color='red', label="Loss")
    ax.plot(acc_list, linestyle='--', color='orange', label='Accuracy')
    ax.plot(rec_list, linestyle='--', color='blue', label='Recall')
    ax.plot(f1_list, linestyle='--', color='green', label='Dice Score')

    legend = ax.legend(loc='upper right', shadow=True)
    legend.get_frame().set_facecolor('#eafff5')
    plt.savefig(join(results_path, f"curves.png"))


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
    elif model_type == "regression":
        criterion = nn.MSELoss()
        mse = torchmetrics.MeanSquaredError()
        psnr = PSNR()
    else:
        raise ValueError("Model type not recognized")

    f1 = DiceScore()

    acc_list = []
    rec_list = []
    f1_list = []

    with tqdm(val_loader, unit="batch") as loader:
        # Iterate over the data
        for data, target in loader:
            loader.set_description("Validation")
            # Move the data to the device
            if model_type == "classification" or model_type == "segmentation":
                data, target = data.to(device), target.to(device)
            elif model_type == "regression":
                data, target = data.to(device), get_noisy_image(target).to(device)

            # Forward pass
            if model_type == "classification":
                output = model(data).logits
                pred = output.argmax(-1)
            else:
                output = model(data)
                pred = output

            f1_score = f1(pred.detach(), target.detach()).numpy()
            f1_list.append(f1_score)

            # Print the loss and metrics
            if model_type == "classification":
                acc = accuracy(pred, target).numpy()
                rec = recall(pred, target).numpy()

                acc_list.append(acc)
                rec_list.append(rec)
            elif model_type == "regression":
                mse_score = mse(output.detach(), target.detach()).numpy()
                val_psnr = psnr(pred.detach(), target.detach()).numpy()

            if criterion is not None:
                # Compute the loss
                loss = criterion(output, target)

                # Print the loss and metrics
                if model_type == "classification":
                    loader.set_postfix(loss=loss.item(), accuracy=acc,
                                       recall=rec,
                                       f1=f1)
                elif model_type == "segmentation":
                    loader.set_postfix(loss=loss.item(), f1=f1_score)
                elif model_type == "regression":
                    loader.set_postfix(loss=loss.item(), mse=mse_score, val_psnr=val_psnr)
            else:
                if model_type == "classification":
                    loader.set_postfix(accuracy=acc_list / len(acc_list),
                                       recall=rec_list / len(rec_list),
                                       f1=f1_list / len(f1_list))
                elif model_type == "segmentation":
                    loader.set_postfix(f1=f1_score)
                elif model_type == "regression":
                    loader.set_postfix(mse=mse_score)

    return np.array(acc_list) / len(acc_list), np.array(rec_list) / len(rec_list), np.array(f1_list) / len(f1_list)


def main(path_to_data, batch_size, epochs, lr, model_name, img_size, results_path, model_type, device):
    """Main function."""

    # Define the data loaders
    _, _, _, num_classes = split_classification_loader(path_to_data, img_size, batch_size)
    if model_type == "classification":
        train_loader, val_loader, test_loader, num_classes = split_classification_loader(path_to_data, img_size,
                                                                                         batch_size)

        for data, target in train_loader:
            print(data.shape)
            break

    elif model_type == "segmentation" or model_type == "regression":
        train_dataset = CustomDataLoader(join(path_to_data, "train/"), img_size, dataset_type="train",
                                         model_type=model_type)
        val_dataset = CustomDataLoader(join(path_to_data, "val/"), img_size, dataset_type="val", model_type=model_type)
        test_dataset = CustomDataLoader(join(path_to_data, "test/"), img_size, dataset_type="test",
                                        model_type=model_type)

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

    dim = target.shape[1]

    # Define the model
    model = build_model(model_name=model_name, model_type=model_type, dim=dim).to(device)

    print(model)

    print(f"Train set size: {len(train_loader.dataset)}")
    print(f"Validation set size: {len(val_loader.dataset)}")
    print(f"Test set size: {len(test_loader.dataset)}")

    # for name, child in model.swin.named_children():
    #     for param in child.parameters():
    #         param.requires_grad = False

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("Data loaded")

    # print(model)

    p = sum([p.numel() for p in model.parameters()])
    print(f"Number of parameters: {p}")

    # Train the model
    print("Training started")
    train(model, train_loader, val_loader, optimizer, epochs, num_classes, device, model_type, results_path)

    # Test the model
    validation(model, test_loader, num_classes, model_type, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Swin Transformer reimplementation")

    parser.add_argument("--path_to_data", type=str, default="./data", help="Path to the data")
    parser.add_argument("--batch_size", "-b", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", "-e", type=int, default=30, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--results", type=str, default="./results/", help="Path to results")
    parser.add_argument("--img_size", type=int, default=256, help="Image size")
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

    print(f"Model type: {model_type}")

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
