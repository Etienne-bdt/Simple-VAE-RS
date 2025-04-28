import torch
import torch.nn.functional as F
import numpy as np

def loss_function(recon_x, x, mu, logvar, gamma):
    # Define the loss function for the VAE
    # Gamma is the variance of the prior
    D = mu.size(1)
    MSE = D*(F.mse_loss(recon_x, x, reduction='mean')/(2*gamma.pow(2)) + (gamma.log()))
    KLD = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - 1 - logvar)
    return MSE , KLD

def cond_loss(recon_x, x, recon_y, y, mu1, logvar1, mu2, logvar2, mu3, logvar3, gamma):
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
    #TODO : verify if gamma is useful and D should be different for the two losses
    D = mu1.size(1)
    MSE_x = D*(F.mse_loss(recon_x, x, reduction='mean')/(2*gamma.pow(2)) + (gamma.log()))
    KLD_u = 0.5 * torch.sum(mu1.pow(2) + logvar1.exp() - 1 - logvar1)
    MSE_y = D*(F.mse_loss(recon_y, y, reduction='mean')/(2*gamma.pow(2)) + (gamma.log()))
    kld_z = 0.5* (torch.sum(logvar3-logvar2) - D +\
            torch.sum((logvar2-logvar3).exp()) +\
            torch.sum((mu2-mu3).pow(2)*((-logvar3).exp())))
    return MSE_x , KLD_u , MSE_y , kld_z
