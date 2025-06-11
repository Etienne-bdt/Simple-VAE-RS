import torch
import torch.nn as nn
import wandb

from loss import base_loss

from .base import BaseVAE
from .layers import down_block, up_block


class VAE(BaseVAE):
    """
    Variational Autoencoder (VAE) model for image reconstruction.
    Inherits from BaseVAE and implements the VAE architecture with an encoder,
    reparameterization trick, and decoder. It also includes methods for training,
    validation, and evaluation.

    Args:
        latent_size (int): Size of the latent space.
        patch_size (int): Size of the input image patches (default: 64).
        callbacks (list, optional): List of callbacks to use during training (default: None).
    """

    def __init__(self, cr, patch_size=64, callbacks=None, slurm_job_id="local"):
        if callbacks is None:
            callbacks = []
        super(VAE, self).__init__(patch_size, callbacks, slurm_job_id)
        self.cr = cr
        self.latent_size = (
            int((patch_size * patch_size * 4 // self.cr) // 4) * 4
        )  # Ensure latent size is a multiple of 4
        self.patch_size = patch_size

        self.gamma = torch.tensor(1.0, requires_grad=True)

        self.encoder = nn.Sequential(
            down_block(in_channels=4, out_channels=64),  # out 16 , 16 , 16
            down_block(in_channels=64, out_channels=256),  # out 64, 8, 8
            down_block(
                in_channels=256,
                out_channels=512,
            ),  # out 512, 4, 4
            down_block(
                in_channels=512,
                out_channels=self.latent_size // 2,
                with_bn=False,
                with_relu=False,
            ),  # out 512, 2, 2,
            nn.Flatten(1),
            # out 512 * 2 * 2 = 2048
        )

        self.decoder = nn.Sequential(
            nn.Unflatten(1, (self.latent_size // 4, 2, 2)),
            up_block(
                in_channels=self.latent_size // 4,
                out_channels=512,
            ),
            up_block(
                in_channels=512,
                out_channels=256,
            ),
            up_block(
                in_channels=256,
                out_channels=64,
            ),
            up_block(
                in_channels=64,
                out_channels=4,
            ),
            nn.Sigmoid(),  # Ensure output is in [0, 1]
        )
        # 4 output channels (same as input)
        self.num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

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
        return self.decoder(z)

    def forward(self, x):
        # Forward pass through the VAE
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def train_step(self, batch, device):
        x, _ = batch
        x = x.to(device)
        x_hat, mu, logvar = self.forward(x)
        mse, kld = base_loss(x_hat, x, mu, logvar, self.gamma)
        loss = mse + kld
        logs = {
            "Loss/loss": loss.item(),
            "Loss/mse": mse.item(),
            "Loss/kld": kld.item(),
        }
        return loss, logs

    def val_step(self, batch, device):
        x, _ = batch
        x = x.to(device)
        with torch.no_grad():
            x_hat, mu, logvar = self.forward(x)
            mse, kld = base_loss(
                x_hat,
                x,
                mu,
                logvar,
                self.gamma,
            )
        loss = mse + kld
        logs = {
            "Loss/val_loss": loss.item(),
            "Loss/val_mse": mse.item(),
            "Loss/val_kld": kld.item(),
        }
        return loss, logs

    def evaluate(self, val_loader, wandb_run, epoch, full_val=False):
        # VAE eval: aggregate SSIM & LPIPS over full validation set
        device = next(self.parameters()).device
        if full_val:
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
                {"Metrics/SSIM": avg_ssim, "Metrics/LPIPS": avg_lpips},
                step=epoch,
            )

        else:
            batch = next(iter(val_loader))
            x, _ = batch
            x = x.to(device)
            with torch.no_grad():
                x_hat, _, _ = self.forward(x)
                imgs_in = x[:4]
                imgs_out = x_hat[:4]

        # log sample images
        if epoch == 1:
            wandb_run.log(
                {
                    "Images/Input": [
                        wandb.Image(img.permute(1, 2, 0).cpu().numpy())
                        for img in imgs_in
                    ],
                },
                step=epoch,
            )
        wandb_run.log(
            {
                "Images/Reconstruction": [
                    wandb.Image(img.permute(1, 2, 0).cpu().numpy()) for img in imgs_out
                ],
            },
            step=epoch,
        )

    def on_train_epoch_end(self, **kwargs):
        self.wandb_run.log(
            {
                "HyperParameters/Gamma": self.gamma.item(),
                "HyperParameters/Learning Rate": self.scheduler.get_last_lr()[0],
            },
            step=self.current_epoch,
        )

    def on_train_start(self, **kwargs):
        self.gamma.requires_grad = True
        self.optimizer.add_param_group({"params": [self.gamma]})

    def get_task_data(self, val_loader):
        batch = next(iter(val_loader))
        x, _ = batch
        x = x.to(next(self.parameters()).device)
        x = x[0:1, :, :, :]
        return x, x

    def sample(self, y, samples):
        """
        Generate samples from the VAE given a condition y.
        Args:
            y (torch.Tensor): Condition tensor of shape (batch_size, latent_size).
            samples (int): Number of samples to generate.
        Returns:
            torch.Tensor: Generated samples of shape (num_samples, 4, patch_size, patch_size).
        """
        mu, logvar = self.encode(y)
        z = torch.randn(samples, self.latent_size, device=y.device)
        z = mu + torch.exp(0.5 * logvar) * z
        return self.decode(z).view(samples, 4, self.patch_size, self.patch_size)


if __name__ == "__main__":
    model = VAE(cr=1.5, patch_size=64)
    print(model)
    y = torch.randn(1, 4, 32, 32)
    y_hat, mu, logvar = model.forward(y)
    print("Output shape:", y_hat.shape)
    print("Mu shape:", mu.shape)
    print("Logvar shape:", logvar.shape)
