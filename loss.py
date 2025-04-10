import torch
import torch.nn.functional as F

def loss_function(recon_x, x, mu, logvar, gamma):
    # Define the loss function for the VAE
    # Gamma is the variance of the prior
    D = mu.size(1)
    MSE = D*(F.mse_loss(recon_x, x, reduction='mean')/(2*gamma.pow(2)) + (gamma.log()))
    KLD = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - 1 - logvar)
    return MSE , KLD
