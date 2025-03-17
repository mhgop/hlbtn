# Pytorch
import torch
from torch import nn
from torchvision.transforms import ToPILImage
# Other
import matplotlib.axes

# --- --- --- Model helpers

def save_model(model:nn.Module, path):
    """Save a model to a file"""
    torch.save(model.state_dict(), path)
    print(f"{model.__class__.__name__} model saved to {path}")


def load_model(model: nn.Module, path):
    """Load a model from a file"""
    model.load_state_dict(torch.load(path, weights_only=True))
    print(f"{model.__class__.__name__} model loaded from {path}")


def print_model_parameters(model):
    """Print the number of parameters of a model"""
    total = 0
    model_name = model.__class__.__name__  # Get the name of the model class
    for name, layer in model.named_modules():
        if hasattr(layer, 'weight') and layer.weight is not None:
            num_params = layer.weight.numel() + (layer.bias.numel() if layer.bias is not None else 0)
            print(f"{model_name}.{name}: {num_params} parameters")
            total += num_params
    print(f"Total parameters: {total}")


# --- --- --- Display helpers

def show_image(ax:matplotlib.axes.Axes, ax_image_size:float, generated_image:torch.Tensor, x:float, y:float, cmap):
    """Display an image on a matplotlib axe"""
    assert generated_image.shape == (1, 28, 28)
    hsize = ax_image_size / 2
    img = ToPILImage()(generated_image)
    ax.imshow(img, vmin=0, vmax=255, aspect='equal', extent=(x - hsize, x + hsize, y - hsize, y + hsize), cmap=cmap)
