import torch
from matplotlib import pyplot as plt

from model import Cond_SRVAE


def test(model: Cond_SRVAE, val_loader):
    """
    Test the model on the validation set and compute error maps for a given image.
    Args:
        model (torch.nn.Module): The trained model.
        val_loader (DataLoader): DataLoader for the validation set
    """
    model.eval()
    batch = next(iter(val_loader))
    y, x = batch
    y, x = y[0, :, :, :].to("cuda"), x[0, :, :, :].to("cuda")

    with torch.no_grad():
        samples = model.sample(y, samples=1000)

    # Compute error map of samples and GT x
    error_map = torch.abs(samples - x).cpu().numpy().mean(axis=0).transpose(1, 2, 0)
    plt.imsave("error_map.png", error_map)
