from torchvision import models


def build_model(model_name):
    """Build a Swin Transformer model based on the model name.
    Args:
        model_name (str): The name of the model to build.
    Returns:
        model (nn.Module): The model.
    """
    if model_name == "tiny":
        model = models.swin_v2_t(weights='IMAGENET1K_V1')
    elif model_name == "small":
        model = models.swin_v2_s(weights='IMAGENET1K_V1')
    elif model_name == "base":
        model = models.swin_v2_b(weights='IMAGENET1K_V1')
    elif model_name == "large":
        model = models.swin_v2_l(weights='IMAGENET1K_V1')
    else:
        raise ValueError("Unknown model name: {}".format(model_name))

    return model
