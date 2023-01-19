import argparse
import math

import timm
import torch
import torch.nn as nn
from transformers import SwinForImageClassification


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class CustomSwinSegModel(nn.Module):
    """Creates a custom model."""

    def __init__(self, model_name, checkpoint, dim=3, upscale_factor=32):
        super(CustomSwinSegModel, self).__init__()
        self.model_name = model_name

        # swin = SwinModel.from_pretrained(checkpoint)
        self.model = timm.create_model('swinv2_tiny_window16_256', pretrained=True, in_chans=dim, num_classes=0)
        # self.model.head = Identity()
        # print(self.model)

        embed_dim = self.model.embed_dim
        num_layers = self.model.num_layers
        num_features = int(embed_dim * 2 ** (num_layers - 1))

        self.decoder = nn.Sequential(
            nn.Conv2d(num_features, num_features * num_layers, kernel_size=(1, 1), stride=(1, 1)),
            nn.PixelShuffle(upscale_factor=upscale_factor))

        if dim == 1:
            in_chans = 3
            self.decoder.add_module('2d', nn.Conv2d(in_chans, 1, kernel_size=(1, 1), stride=(1, 1)))

    def forward(self, x):
        x = self.model(x)

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
        # model = SwinForMaskedImageModeling.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
        # model.load_state_dict(torch.load("results/model_denoising_1500.pth"))
    elif model_type == "classification":
        model = SwinForImageClassification.from_pretrained(cp)
    else:
        raise ValueError("Invalid model type")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model builder")
    parser.add_argument("--debug", "-d", action="store_true")
    args = parser.parse_args()

    debug = args.debug

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    print(f"Used devide : {device}")

    dim = 1

    if dim == 1:
        x = torch.rand(1, 1, 256, 256).to(device)
    else:
        x = torch.rand(1, 3, 256, 256).to(device)

    if debug:
        activation = {}


        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()

            return hook


        model = timm.create_model('swinv2_tiny_window16_256', pretrained=True, in_chans=dim, num_classes=0)

        print(model)

        model.layers[3].blocks[0].register_forward_hook(get_activation('timm_layer_3_block_0'))
        model.layers[3].blocks[1].attn.register_forward_hook(get_activation('timm_layer_3_block_1_attn'))
        model.layers[3].blocks[1].norm1.register_forward_hook(get_activation('timm_layer_3_block_1_norm1'))
        model.layers[3].blocks[1].mlp.register_forward_hook(get_activation('timm_layer_3_block_1_mlp'))
        model.layers[3].blocks[1].norm2.register_forward_hook(get_activation('timm_layer_3_block_1_norm2'))
        model.layers[3].blocks[1].drop_path2.register_forward_hook(get_activation('timm_layer_3_block_1_drop_path2'))
        model.layers[3].downsample.register_forward_hook(get_activation('timm_layer_3_block_2_attn'))
        model.norm.register_forward_hook(get_activation('timm_norm'))
        model.head.register_forward_hook(get_activation('timm_head'))

        _ = model(x)

        print("Output shapes:")
        for k, _ in activation.items():
            print(f"Block {k}: {activation[k].shape}")

    else:
        model = build_model(model_name="tiny", model_type="segmentation", dim=dim)
        print(model)

    p = sum([p.numel() for p in model.parameters()])
    print(f"Number of parameters: {p}")

    if dim == 1:
        x = torch.rand(1, 1, 256, 256).to(device)
    else:
        x = torch.rand(1, 3, 256, 256).to(device)

    output = model(x.to(device))
    print(f"Output shape: {output.shape}")
