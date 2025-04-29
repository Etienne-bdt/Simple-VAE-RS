import torch
import torch.nn.functional as F


def loss_function(recon_x, x, mu, logvar, gamma):
    # Define the loss function for the VAE
    # Gamma is the variance of the prior
    d = mu.size(1)
    mse = d * (
        F.mse_loss(recon_x, x, reduction="mean") / (2 * gamma.pow(2)) + (gamma.log())
    )
    kld = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - 1 - logvar)
    return mse, kld


def cond_loss(
    recon_x, x, recon_y, y, mu1, logvar1, mu2, logvar2, mu3, logvar3, gamma, gamma2
):
    """
    Conditional loss function for the VAE
    recon_x: reconstructed HR image
    x: original HR image
    recon_y: reconstructed LR image
    y: original LR image
    mu1: mean of the latent variable for the LR image
    logvar1: log variance of the latent variable for the LR image
    mu2: mean of the latent variable for the HR image depending on the HR image
    logvar2: log variance of the latent variable for the HR image depending on the HR image
    mu3: mean of the latent variable for the HR image depending on the LR image
    logvar3: log variance of the latent variable for the HR image depending on the LR image
    gamma: variance of the prior

    Note: Parameters 1 and 2,3 represent respectively the latent variables of the u and z.
    2 : z from x
    3 : z from y,u
    """
    d = mu1.size(1)
    mse_x = d * (
        F.mse_loss(recon_x, x, reduction="mean") / (2 * gamma.pow(2)) + (gamma.log())
    )
    kld_u = 0.5 * torch.sum(mu1.pow(2) + logvar1.exp() - 1 - logvar1)
    mse_y = d * (
        F.mse_loss(recon_y, y, reduction="mean") / (2 * gamma2.pow(2)) + (gamma2.log())
    )
    kld_z = 0.5 * (
        torch.sum(logvar3 - logvar2)
        - d
        + torch.sum((logvar2 - logvar3).exp())
        + torch.sum((mu2 - mu3).pow(2) * ((-logvar3).exp()))
    )
    return mse_x, kld_u, mse_y, kld_z
