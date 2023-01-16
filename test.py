import argparse
import os
from os.path import join

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from skimage.metrics import peak_signal_noise_ratio as PSNR

from dataloader import get_noisy_image, split_loader, transformations
from model import build_model


def evaluate(val_image, model, loss_fn, noise_parameter, device):
    model.eval()

    with torch.no_grad():
        data = get_noisy_image(val_image, noise_parameter)
        data = data.to(device)

        preds = model(data)

        target = torch.clone(val_image)
        target = target.to(device)

    psnr = PSNR(target.cpu().detach().numpy()[0], preds.cpu().detach().numpy()[0])
    mse = loss_fn(target.cpu().detach(), preds.cpu().detach())

    return psnr, mse


def train(train_image, val_image, model, optimizer, loss_fn, epochs, device, noise_parameter, results_path,
          color_space):
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    data = get_noisy_image(train_image, noise_parameter)
    data = data.to(device)

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs} | ", end="")
        for _ in range(10):
            preds = model(data)
            target = torch.clone(train_image)
            target = target.to(device)

            loss = loss_fn(preds, target)
            train_psnr = PSNR(target.cpu().detach().numpy()[0], preds.cpu().detach().numpy()[0])
            train_psnr = train_psnr.mean()

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        psnr, mse = evaluate(val_image, model, loss_fn, noise_parameter, device)
        print(f"train_psnr: {train_psnr:.3f} - train_mse: {loss:.5f}", end="")
        print(f" - val_psnr: {psnr:.3f} - val_mse: {mse:.5f}")

        if epoch % 10 == 0:
            torch.save(model.state_dict(), join(results_path, f"model_{color_space}.pth"))

    torch.save(model.state_dict(), join(results_path, f"model_{color_space}.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--path_to_data", type=str, default="./data", help="Path to the data")
    parser.add_argument("--epochs", "-e", type=int, default=30)
    parser.add_argument("--learning_rate", "-l", type=float, default=1e-3)
    parser.add_argument("--batch_size", "-b", type=int, default=1)
    parser.add_argument("--noise_parameter", "-n", type=float, default=0.2)
    parser.add_argument("--results", "-r", type=str, default="results/")
    parser.add_argument("--img_size", type=int, default=224, help="Image size")
    parser.add_argument("--model_name", type=str, default="STUnet", help="Model name")
    parser.add_argument("--loader", action='store_true')
    parser.add_argument("--load", action='store_true')

    args = parser.parse_args()

    path_to_data = args.path_to_data
    num_epochs = args.epochs
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    noise_parameter = args.noise_parameter
    img_size = args.img_size
    results_path = args.results
    model_name = args.model_name
    loader = args.loader
    transfert_learning = args.load

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "cpu"
    else:
        device = "cpu"

    print(f"Using device: {device}")

    if loader:
        train_loader, val_loader, test_loader, num_classes = split_loader(path_to_data, img_size, batch_size)
        print("Data loaded")

        inputs, _ = next(iter(train_loader))

    else:
        train_image = cv2.imread(join(path_to_data, "DSC00213.jpg"))
        val_image = cv2.imread(join(path_to_data, "DSC00213.jpg"))
        test_image = cv2.imread(join(path_to_data, "DSC00213.jpg"))

        transform = transformations(img_size)

        train_image = transform(train_image).unsqueeze(0)
        val_image = transform(val_image).unsqueeze(0)
        test_image = transform(test_image).unsqueeze(0)

        print("Data loaded")

        inputs = train_image

    chan = inputs.shape[1]
    color_space = "rgb" if chan == 3 else "gray"

    model = build_model(model_name)
    model = model.to(device)

    if transfert_learning:
        model.load_state_dict(torch.load(join(results_path, f"model_{color_space}.pth")))
        print("Model loaded")

    p = 0
    for param in model.parameters():
        p += param.numel()
    print(f"Number of parameters: {p}")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    # Create the results directory
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    train(train_image, val_image,
          model, optimizer, loss_fn,
          num_epochs, device, noise_parameter,
          results_path, color_space)

    psnr, mse = evaluate(test_image, model, loss_fn, noise_parameter, device)
    print(f"test_psnr : {psnr:.3f} - test_mse : {mse:.5f}")
