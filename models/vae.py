import torch
import torch.nn as nn

from loss import base_loss

from .base import BaseVAE


class VAE(BaseVAE):
    def __init__(self, latent_size):
        super(VAE, self).__init__()
        self.latent_size = latent_size

        self.gamma = torch.tensor(1.0, requires_grad=True)

        self.encoder = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1),  # 4 input channels (
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 32 input channels
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 64 input channels
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(128 * 8 * 8 * 16, self.latent_size)
        self.fc_logvar = nn.Linear(128 * 8 * 8 * 16, self.latent_size)
        self.fc_decode = nn.Linear(self.latent_size, 128 * 8 * 8 * 16)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                128, 64, kernel_size=3, stride=2, padding=1
            ),  # 128 input channels
            nn.ReLU(),
            nn.ConvTranspose2d(
                64, 32, kernel_size=4, stride=2, padding=1
            ),  # 64 input channels
            nn.ReLU(),
            nn.ConvTranspose2d(
                32, 4, kernel_size=8, stride=2, padding=1
            ),  # 32 input channels
            nn.Sigmoid(),
        )
        # 4 output channels (same as input)

    def encode(self, x):
        # Define the encoder part of the VAE
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = self.fc_decode(z)
        z = z.view(z.size(0), 128, 8 * 4, 8 * 4)
        x = self.decoder(z)
        return x

    def forward(self, x):
        # Forward pass through the VAE
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def train_step(self, batch, device, loss_fn):
        x = batch.to(device)
        x_hat, mu, logvar = self(x)
        loss, kld = base_loss(x_hat, x, mu, logvar, self.gamma)
        logs = {"loss": loss.item(), "kld": kld.item()}
        return loss, logs

    def val_step(self, batch, device, loss_fn):
        x = batch.to(device)
        with torch.no_grad():
            x_hat, mu, logvar = self(x)
            loss, kld = base_loss(
                x_hat,
                x,
                mu,
                logvar,
                self.gamma,
            )
        logs = {"val_loss": loss.item(), "val_kld": kld.item()}
        return loss, logs
