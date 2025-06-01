import torch
import torch.nn as nn
import wandb

from loss import base_loss

from .base import BaseVAE
from .layers import downsample_sequence, upsample_sequence


class VAE(BaseVAE):
    def __init__(self, latent_size, patch_size=64, callbacks=None):
        if callbacks is None:
            callbacks = []
        super(VAE, self).__init__(patch_size, callbacks)
        self.latent_size = latent_size
        self.patch_size = patch_size

        self.gamma = torch.tensor(1.0, requires_grad=True)

        self.encoder = downsample_sequence(
            in_shape=(4, patch_size, patch_size),
            out_flattened_size=latent_size*2,
            out_channels=256,
            num_steps=5
        )
        self.decoder = upsample_sequence(
            in_channels=128,
            in_flattened_size=latent_size,
            out_shape=(4, patch_size, patch_size),
            num_steps=5
        )
        # 4 output channels (same as input)

    def encode(self, x):
        # Define the encoder part of the VAE
        x = self.encoder(x)
        return x.chunk(2, dim=1)  # Split into mu and logvar

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

    def train_step(self, batch, device):
        x, _ = batch
        x = x.to(device)
        x_hat, mu, logvar = self.forward(x)
        loss, kld = base_loss(x_hat, x, mu, logvar, self.gamma)
        logs = {"loss": loss.item(), "kld": kld.item()}
        return loss, logs

    def val_step(self, batch, device):
        x, _ = batch
        x = x.to(device)
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

        for batch in val_loader:
            x, _ = batch
            x = x.to(device)
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
        self.gamma.requires_grad = True
        self.optimizer.add_param_group({"params": [self.gamma]})
