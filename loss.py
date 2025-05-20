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
    Loss function for the Conditional Super-Resolution VAE. This function minimizes the **NEGATIVE** ELBO:
    .. math::
        -(E[log p(x|z)] - D_KL(q(z|x)||p(z))) .

    Args:
        recon_x (torch.Tensor): Reconstructed high-resolution (HR) image.
        x (torch.Tensor): Original high-resolution (HR) image.
        recon_y (torch.Tensor): Reconstructed low-resolution (LR) image.
        y (torch.Tensor): Original low-resolution (LR) image.
        mu1 (torch.Tensor): Mean of the latent variable for the LR image.
        logvar1 (torch.Tensor): Log variance of the latent variable for the LR image.
        mu2 (torch.Tensor): Mean of the latent variable for the HR image conditioned on the HR image.
        logvar2 (torch.Tensor): Log variance of the latent variable for the HR image conditioned on the HR image.
        mu3 (torch.Tensor): Mean of the latent variable for the HR image conditioned on the LR image.
        logvar3 (torch.Tensor): Log variance of the latent variable for the HR image conditioned on the LR image.
        gamma (torch.Tensor): Variance of the prior for the HR image.
        gamma2 (torch.Tensor): Variance of the prior for the LR image.

    Returns:
        tuple: A tuple containing:
            - mse_x (torch.Tensor): Mean squared error for the HR image.
            - kld_u (torch.Tensor): Kullback-Leibler divergence for the LR image.
            - mse_y (torch.Tensor): Mean squared error for the LR image.
            - kld_z (torch.Tensor): Kullback-Leibler divergence for the HR image.

    Note:
        Parameters 1 (mu1, logvar1) and 2,3 (mu2, logvar2, mu3, logvar3) represent the latent variables for `u` and `z`, respectively:
        - 2: `z` derived from `x`.
        - 3: `z` derived from `y` and `u`.
    """
    y_shape = recon_y.shape
    x_shape = recon_x.shape
    n_y = y_shape[0] * y_shape[1] * y_shape[2] * y_shape[3]
    n_x = x_shape[0] * x_shape[1] * x_shape[2] * x_shape[3]
    mse_y = n_y * (
        F.mse_loss(recon_y, y, reduction="mean") / (2 * gamma2.pow(2)) + (gamma2.log())
    )
    kld_u = 0.5 * torch.sum(mu1.pow(2) + logvar1.exp() - 1 - logvar1, dim=1).mean()
    mse_x = n_x * (
        F.mse_loss(recon_x, x, reduction="mean") / (2 * gamma.pow(2)) + (gamma.log())
    )
    kld_z = (
        0.5
        * (
            torch.sum(logvar3 - logvar2 - 1, dim=1)
            + torch.sum((logvar2 - logvar3).exp(), dim=1)
            + torch.sum((mu2 - mu3).pow(2) * ((-logvar3).exp()), dim=1)
        ).mean()
    )
    return mse_x, kld_u, mse_y, kld_z


def cond_loss_mog(
    recon_x,
    x,
    recon_y,
    y,
    mu1,
    logvar1,
    mu2,
    logvar2,
    mu3,
    logvar3,
    gamma,
    gamma2,
    weight,
    means,
    logvars,
):
    """
    Loss function for the Conditional Super-Resolution VAE. This function minimizes the **NEGATIVE** ELBO:
    .. math::
        -(E[log p(x|z)] - D_KL(q(z|x)||p(z))) .

    Args:
        recon_x (torch.Tensor): Reconstructed high-resolution (HR) image.
        x (torch.Tensor): Original high-resolution (HR) image.
        recon_y (torch.Tensor): Reconstructed low-resolution (LR) image.
        y (torch.Tensor): Original low-resolution (LR) image.
        mu1 (torch.Tensor): Mean of the latent variable for the LR image.
        logvar1 (torch.Tensor): Log variance of the latent variable for the LR image.
        mu2 (torch.Tensor): Mean of the latent variable for the HR image conditioned on the HR image.
        logvar2 (torch.Tensor): Log variance of the latent variable for the HR image conditioned on the HR image.
        mu3 (torch.Tensor): Mean of the latent variable for the HR image conditioned on the LR image.
        logvar3 (torch.Tensor): Log variance of the latent variable for the HR image conditioned on the LR image.
        gamma (torch.Tensor): Variance of the prior for the HR image.
        gamma2 (torch.Tensor): Variance of the prior for the LR image.

    Returns:
        tuple: A tuple containing:
            - mse_x (torch.Tensor): Mean squared error for the HR image.
            - kld_u (torch.Tensor): Kullback-Leibler divergence for the LR image.
            - mse_y (torch.Tensor): Mean squared error for the LR image.
            - kld_z (torch.Tensor): Kullback-Leibler divergence for the HR image.

    Note:
        Parameters 1 (mu1, logvar1) and 2,3 (mu2, logvar2, mu3, logvar3) represent the latent variables for `u` and `z`, respectively:
        - 2: `z` derived from `x`.
        - 3: `z` derived from `y` and `u`.
    """
    y_shape = recon_y.shape
    x_shape = recon_x.shape
    n_y = y_shape[0] * y_shape[1] * y_shape[2] * y_shape[3]
    n_x = x_shape[0] * x_shape[1] * x_shape[2] * x_shape[3]
    mse_y = n_y * (
        F.mse_loss(recon_y, y, reduction="mean") / (2 * gamma2.pow(2)) + (gamma2.log())
    )

    # Compute KL divergence for each batch element against the mixture of Gaussians
    batch_size, latent_dim = mu1.size()
    num_components = means.size(0)

    # Expand dimensions for broadcasting
    mu1_exp = mu1.unsqueeze(1).expand(-1, num_components, -1)
    logvar1_exp = logvar1.unsqueeze(1).expand(-1, num_components, -1)
    means_exp = means.unsqueeze(0).expand(batch_size, -1, -1)
    logvars_exp = logvars.unsqueeze(0).expand(batch_size, -1, -1)
    weight_exp = weight.unsqueeze(0).expand(batch_size, -1)

    # Compute KL divergence for all components
    kld_u_components = 0.5 * (
        torch.sum(logvars_exp - logvar1_exp - 1, dim=2)
        + torch.sum((logvar1_exp - logvars_exp).exp(), dim=2)
        + torch.sum((mu1_exp - means_exp).pow(2) * (-logvars_exp).exp(), dim=2)
    )

    # Weight the components and sum
    kld_u_mog = torch.sum(weight_exp * kld_u_components, dim=1).mean()

    mse_x = n_x * (
        F.mse_loss(recon_x, x, reduction="mean") / (2 * gamma.pow(2)) + (gamma.log())
    )
    kld_z = (
        0.5
        * (
            torch.sum(logvar3 - logvar2 - 1, dim=1)
            + torch.sum((logvar2 - logvar3).exp(), dim=1)
            + torch.sum((mu2 - mu3).pow(2) * ((-logvar3).exp()), dim=1)
        ).mean()
    )
    return mse_x, kld_u_mog, mse_y, kld_z
