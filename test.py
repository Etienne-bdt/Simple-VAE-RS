import torch
from matplotlib import pyplot as plt

from model import Cond_SRVAE


def test(device, model: Cond_SRVAE, val_loader):
    """
    Test the model on the validation set and compute error maps for a given image.
    Args:
        model (torch.nn.Module): The trained model.
        val_loader (DataLoader): DataLoader for the validation set
    """
    model.eval()
    batch = next(iter(val_loader))
    y, x = batch
    y, x = y[0:1, :, :, :].to(device), x[0:1, :, :, :].to(device)

    with torch.no_grad():
        samples = model.sample(y, samples=1000)

    # Compute error map of samples and GT x
    diff = (samples - x).mean(dim=1)
    error_map = torch.abs(diff).cpu().numpy().mean(axis=0)
    plt.figure(figsize=(40, 40))
    plt.subplot(2, 2, 1)
    plt.imshow(y[0, [2, 1, 0], :, :].cpu().numpy().transpose(1, 2, 0))
    plt.title("Input Image")
    plt.subplot(2, 2, 2)
    plt.imshow(samples[0, [2, 1, 0], :, :].cpu().numpy().transpose(1, 2, 0))
    plt.title("Sampled Image")
    plt.subplot(2, 2, 3)
    plt.imshow(error_map, cmap="hot")
    plt.colorbar()
    plt.title("Error Map")
    plt.subplot(2, 2, 4)
    var = diff.var(dim=0).cpu().numpy()
    plt.imshow(var, cmap="hot")
    plt.colorbar()
    plt.title("Variance Map")
    plt.savefig("variance_map_with_title.png", bbox_inches="tight")
    plt.close()
