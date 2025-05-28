import os
import time

import torch
from matplotlib import pyplot as plt

import models


def test(device, model: models.Cond_SRVAE, val_loader):
    """
    Test the model on the validation set and compute error maps for a given image.
    Args:
        model (torch.nn.Module): The trained model.
        val_loader (DataLoader): DataLoader for the validation set
    """
    slurm_job_id = os.environ.get(
        "SLURM_JOB_ID", f"local_{time.strftime('%Y%m%D-%H%M%S')}"
    )
    os.makedirs(os.path.join("results", slurm_job_id), exist_ok=True)
    model.eval()
    batch = next(iter(val_loader))
    y, x = batch
    y, x = y[1:2, :, :, :].to(device), x[1:2, :, :, :].to(device)

    with torch.no_grad():
        samples = model.sample(y, samples=1000)

    # Compute error map of samples and GT x
    diff = samples - x
    mean = samples.mean(dim=0).cpu().numpy()
    std = samples.std(dim=0).cpu().numpy().mean(axis=0)
    mae = diff.abs().mean(dim=(0, 1)).cpu().numpy()
    mse = (diff.pow(2)).mean(dim=(0, 1)).cpu().numpy()

    plt.figure(figsize=(20, 10))
    plt.subplot(2, 4, 1)
    plt.imshow(y[0, [2, 1, 0], :, :].cpu().numpy().transpose(1, 2, 0))
    plt.title("Input Image")
    plt.subplot(2, 4, 2)
    plt.imshow(samples[0, [2, 1, 0], :, :].cpu().numpy().transpose(1, 2, 0))
    plt.title("Sampled Image")
    plt.subplot(2, 4, 3)
    plt.imshow(x[0, [2, 1, 0], :, :].cpu().numpy().transpose(1, 2, 0))
    plt.title("Ground Truth Image")
    plt.subplot(2, 4, 4)
    plt.imshow(mean[[2, 1, 0], :, :].transpose(1, 2, 0))
    plt.title("Mean of Samples")
    plt.subplot(2, 4, 5)
    plt.imshow(mae, cmap="hot")
    plt.colorbar()
    plt.title("MAE Map")
    plt.subplot(2, 4, 6)
    plt.imshow(mse, cmap="hot")
    plt.colorbar()
    plt.title("MSE Map")
    plt.subplot(2, 4, 7)
    plt.imshow(std, cmap="hot")
    plt.colorbar()
    plt.title(f"Standard Deviation of Samples, Mean: {std.mean():.2f}")
    plt.subplot(2, 4, 8)
    mean_bias = (x - samples.mean(dim=0)).mean(dim=0).cpu().numpy()
    plt.imshow(mean_bias, cmap="hot")
    plt.colorbar()
    plt.title(f"Mean Bias Map, Mean: {mean_bias.mean():.2f}")
    plt.savefig(f"results/{slurm_job_id}/error_mean_std_maps.png", bbox_inches="tight")
    plt.close()
    MMSE = (samples - x).pow(2).mean()
    print(f"MMSE: {MMSE:.4f}")

    plt.figure(figsize=(10, 10))
    with torch.no_grad():
        y_gen, x_gen = model.generation()
    plt.subplot(2, 1, 1)
    plt.imshow(y_gen[0, [2, 1, 0], :, :].detach().cpu().numpy().transpose(1, 2, 0))
    plt.title("Generated Image")
    plt.subplot(2, 1, 2)
    plt.imshow(x_gen[0, [2, 1, 0], :, :].detach().cpu().numpy().transpose(1, 2, 0))
    plt.title("Generated Image from x")
    plt.savefig(f"results/{slurm_job_id}/generated_image.png", bbox_inches="tight")
    plt.close()
