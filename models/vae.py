import torch
import torch.nn as nn

import wandb
from loss import base_loss

from .base import BaseVAE


class VAE(BaseVAE):
    def __init__(self, latent_size, patch_size=64):
        super(VAE, self).__init__(patch_size)
        self.latent_size = latent_size
        self.patch_size = patch_size

        self.gamma = torch.tensor(1.0, requires_grad=True)

        self.encoder = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1),  # 4 input channels (
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 32 input channels
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 64 input channels
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(patch_size**2 // 2, self.latent_size)
        self.fc_logvar = nn.Linear(patch_size**2 // 2, self.latent_size)
        self.fc_decode = nn.Linear(self.latent_size, patch_size**2 // 2)

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
        z = z.view(z.size(0), 128, self.patch_size // 16, self.patch_size // 16)
        x = self.decoder(z)
        return x

    def forward(self, x):
        # Forward pass through the VAE
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def train_step(self, batch, device, loss_fn):
        x = batch.to(device)
        x_hat, mu, logvar = self.forward(x)
        loss, kld = base_loss(x_hat, x, mu, logvar, self.gamma)
        logs = {"loss": loss.item(), "kld": kld.item()}
        return loss, logs

    def val_step(self, batch, device, loss_fn):
        x = batch.to(device)
        with torch.no_grad():
            x_hat, mu, logvar = self.forward(x)
            loss, kld = base_loss(
                x_hat,
                x,
                mu,
                logvar,
                self.gamma,
            )
        logs = {"val_loss": loss.item(), "val_kld": kld.item()}
        return loss, logs

    def evaluate(self, val_loader, wandb_run, epoch):
        # VAE eval: aggregate SSIM & LPIPS over full validation set

        device = next(self.parameters()).device

        total_pixels = 0
        total_ssim = 0.0
        total_lpips = 0.0
        first_batch = True

        for xb in val_loader:
            x = xb.to(device)
            with torch.no_grad():
                x_hat, _, _ = self.forward(x)

            b = x.size(0)
            total_pixels += b

            # per-sample metrics
            for orig, recon in zip(x, x_hat):
                ssim = self.ssim(
                    orig.cpu().numpy(),
                    recon.cpu().numpy(),
                    win_size=11,
                    data_range=1.0,
                    channel_axis=0,
                )
                total_ssim += ssim
                total_lpips += self.lpips_fn(
                    orig[[2, 1, 0]].unsqueeze(0), recon[[2, 1, 0]].unsqueeze(0)
                ).item()

            # capture first batch for image logging
            if first_batch:
                imgs_in = x[:4]
                imgs_out = x_hat[:4]
                first_batch = False

        # compute averages
        avg_ssim = total_ssim / total_pixels
        avg_lpips = total_lpips / total_pixels

        # log aggregate metrics
        wandb_run.log(
            {"VAE/SSIM": avg_ssim, "VAE/LPIPS": avg_lpips},
            step=epoch,
        )

        # log sample images
        wandb_run.log(
            {
                "VAE/Input": [
                    wandb.Image(img.permute(1, 2, 0).cpu().numpy()) for img in imgs_in
                ],
                "VAE/Reconstruction": [
                    wandb.Image(img.permute(1, 2, 0).cpu().numpy()) for img in imgs_out
                ],
            },
            step=epoch,
        )

    def on_train_epoch_end(self, **kwargs):
        self.wandb_run.log(
            {
                "VAE/Gamma": self.gamma.item(),
                "VAE/Learning Rate": self.scheduler.get_last_lr()[0],
            },
            step=self.current_epoch,
        )

    def on_train_start(self, **kwargs):
        return super().on_train_start(**kwargs)
