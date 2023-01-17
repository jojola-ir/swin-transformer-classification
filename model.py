import math

import torch
import torch.nn as nn
from transformers import SwinModel, SwinForImageClassification


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class CustomSwinSegModel(nn.Module):
    """Creates a custom model."""

    def __init__(self, model_name, checkpoint, dim=3, upscale_factor=32):
        super(CustomSwinSegModel, self).__init__()
        self.model_name = model_name

        swin = SwinModel.from_pretrained(checkpoint)
        swin.pooler = Identity()

        embed_dim = swin.config.hidden_size
        num_layers = swin.config.num_hidden_layers
        num_features = int(embed_dim * 2 ** (num_layers - 2))

        self.model = swin

        self.decoder = nn.Sequential(nn.Conv2d(embed_dim, num_features, kernel_size=(1, 1), stride=(1, 1)),
                                     nn.PixelShuffle(upscale_factor=upscale_factor),
                                     nn.Conv2d(3, dim, kernel_size=(1, 1), stride=(1, 1)))

    def forward(self, x):
        x = self.model(x).last_hidden_state

        x = x.transpose(1, 2)
        batch_size, num_channels, sequence_length = x.shape
        height = width = math.floor(sequence_length ** 0.5)
        x = x.reshape(batch_size, num_channels, height, width)

        x = self.decoder(x)

        return x


def build_model(model_name, model_type, dim):
    """Build a tiny Swin-Transformer.
    Args:
        model_name (str): The name of the model to build.
    Returns:
        model (nn.Module): The model.
    """
    cp = "microsoft/swin-tiny-patch4-window7-224"
    if model_type == "segmentation" or model_type == "regression":
        model = CustomSwinSegModel(model_name, checkpoint=cp, dim=dim)
    elif model_type == "classification":
        model = SwinForImageClassification.from_pretrained(cp)
    else:
        raise ValueError("Invalid model type")

    return model


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    print(f"Used devide : {device}")

    model = build_model(model_name="tiny", model_type="segmentation", dim=1)
    model.to(device)

    print(model)

    p = sum([p.numel() for p in model.parameters()])
    print(f"Number of parameters: {p}")

    x = torch.randn((1, 3, 224, 224))
    output = model(x.to(device))
    print(f"Output shape: {output.shape}")
