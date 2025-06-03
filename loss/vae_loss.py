import torch
import torch.nn.functional as F


def base_loss(recon_x, x, mu, logvar, gamma):
    # Define the loss function for the VAE
    # Gamma is the variance of the prior
    d = recon_x.shape[0] * recon_x.shape[1] * recon_x.shape[2] * recon_x.shape[3]
    mse = d * (
        F.mse_loss(recon_x, x, reduction="mean") / (2 * gamma.pow(2)) + (gamma.log())
    )
    kld = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - 1 - logvar, dim=1).mean()
    return mse, kld
