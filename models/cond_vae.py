import os

import torch
import torch.nn as nn
import wandb
from torch.nn import functional as F
from tqdm import tqdm

from loss import cond_loss

from .base import BaseVAE
from .layers import downsample_sequence, upsample_sequence


class Cond_SRVAE(BaseVAE):
    def __init__(self, latent_size, patch_size=64, callbacks=None):
        if callbacks is None:
            callbacks = []
        super(Cond_SRVAE, self).__init__(patch_size, callbacks)
        self.latent_size = latent_size
        self.latent_size_y = latent_size // 4
        self.patch_size = patch_size
        self.gammax = torch.tensor(1.0, requires_grad=True)
        self.gammay = torch.tensor(1.0, requires_grad=True)

        self.encoder_y = downsample_sequence(
            in_shape=(4, int(patch_size // 2), int(patch_size // 2)),
            out_flattened_size=self.latent_size_y * 2,
            out_channels=256,
            num_steps=10,
        )

        self.decoder_y = upsample_sequence(
            in_flattened_size=(self.latent_size_y),
            out_shape=(4, patch_size / 2, patch_size / 2),
            num_steps=10,
            in_channels=128,
        )

        self.encoder_x = downsample_sequence(
            in_shape=(4, patch_size, patch_size),
            out_flattened_size=self.latent_size * 2,
            out_channels=256,
            num_steps=10,
        )

        self.decoder_x = upsample_sequence(
            in_channels=256,
            in_flattened_size=(self.latent_size * 2),
            out_shape=(4, patch_size, patch_size),
            num_steps=10,
        )

        self.y_to_z = downsample_sequence(
            in_shape=(4, patch_size // 2, patch_size // 2),
            out_flattened_size=self.latent_size,
            out_channels=128,
            num_steps=10,
        )
        # Replace Linear layers with Conv-based alternatives
        # u_to_z: expects input of shape (batch, latent_size_y)
        # We'll reshape to (batch, channels, 1, 1) and use a 1x1 Conv
        self.u_to_z = nn.Sequential(
            nn.Unflatten(1, (self.latent_size_y, 1, 1)),
            nn.Conv2d(self.latent_size_y, self.latent_size_y * 2, kernel_size=1),
            nn.Conv2d(self.latent_size_y * 2, self.latent_size, kernel_size=1),
            nn.Conv2d(self.latent_size, self.latent_size, kernel_size=1),
            nn.Flatten(1),
        )
        # mu_u_y_to_z and logvar_u_y_to_z: input is (batch, latent_size*2)
        # We'll reshape to (batch, latent_size*2, 1, 1) and use 1x1 Conv
        self.mu_u_y_to_z = nn.Sequential(
            nn.Unflatten(1, (self.latent_size * 2, 1, 1)),
            nn.Conv2d(self.latent_size * 2, self.latent_size * 2, kernel_size=1),
            nn.Conv2d(self.latent_size * 2, self.latent_size * 2, kernel_size=1),
            nn.Conv2d(self.latent_size * 2, self.latent_size, kernel_size=1),
            nn.Flatten(1),
        )
        self.logvar_u_y_to_z = nn.Sequential(
            nn.Unflatten(1, (self.latent_size * 2, 1, 1)),
            nn.Conv2d(self.latent_size * 2, self.latent_size * 2, kernel_size=1),
            nn.Conv2d(self.latent_size * 2, self.latent_size * 2, kernel_size=1),
            nn.Conv2d(self.latent_size * 2, self.latent_size, kernel_size=1),
            nn.Flatten(1),
            nn.Hardtanh(-7, 7),
        )

        self.num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def z_cond(self, y, u):
        # Define the encoder part of the VAE
        y = self.y_to_z(y)
        y = y.view(y.size(0), -1)

        u = self.u_to_z(u)
        u = u.view(u.size(0), -1)

        jointure = torch.cat((y, u), dim=1)

        mu_u_y = self.mu_u_y_to_z(jointure)
        logvar_u_y = self.logvar_u_y_to_z(jointure)

        return mu_u_y, logvar_u_y

    def encode_y(self, y):
        # Define the encoder part of the VAE
        y = self.encoder_y(y)
        return torch.chunk(y, 2, dim=1)

    def encode_x(self, x):
        # Define the encoder part of the VAE
        x = self.encoder_x(x)
        return torch.chunk(x, 2, dim=1)

    def reparameterize(self, mu, logvar):
        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode_y(self, u):
        return self.decoder_y(u)

    def decode_x(self, z, y):
        y_enc = self.y_to_z(y)
        stack = torch.cat((y_enc, z), dim=1)
        return self.decoder_x(stack)

    def forward(self, x, y):
        mu_u, logvar_u = self.encode_y(y)
        u = self.reparameterize(mu_u, logvar_u)

        mu_z, logvar_z = self.encode_x(x)
        z = self.reparameterize(mu_z, logvar_z)

        mu_z_uy, logvar_z_uy = self.z_cond(y, u)

        x_hat = self.decode_x(z, y)
        y_hat = self.decode_y(u)

        return x_hat, y_hat, mu_z, logvar_z, mu_u, logvar_u, mu_z_uy, logvar_z_uy

    def conditional_generation(self, y):
        # Generate a sample from the model
        mu_u, logvar_u = self.encode_y(y)
        u = self.reparameterize(mu_u, logvar_u)

        mu_z_uy, logvar_z_uy = self.z_cond(y, u)
        z = self.reparameterize(mu_z_uy, logvar_z_uy)

        x_hat = self.decode_x(z, y)
        return x_hat

    def sample(self, y, samples=1000) -> torch.Tensor:
        # Generate samples from the model
        mu_u, logvar_u = self.encode_y(y)
        u = self.reparameterize(mu_u, logvar_u)

        mu_z_uy, logvar_z_uy = self.z_cond(y, u)

        std = torch.exp(0.5 * logvar_z_uy)
        latent = std.size(1)
        eps = torch.randn((samples, latent)).to(y.device)

        z = mu_z_uy + eps * std

        if y.ndim == 3:
            y = y.unsqueeze(0)
            # Using expand to not reallocate for the same tensor
            y = y.expand(z.size(0), -1, -1, -1)
        elif y.ndim == 4 and y.size(0) == 1:
            y = y.expand(z.size(0), -1, -1, -1)
        return self.decode_x(z, y)

    def generation(self):
        u = torch.randn(1, self.latent_size_y).to("cuda")
        y = self.decode_y(u)

        return y, self.conditional_generation(y)

    def train_step(self, batch, device):
        y, x = batch
        y, x = y.to(device), x.to(device)
        x_hat, y_hat, mu_z, logvar_z, mu_u, logvar_u, mu_z_uy, logvar_z_uy = (
            self.forward(x, y)
        )
        mse_x, kld_u, mse_y, kld_z = cond_loss(
            x_hat,
            x,
            y_hat,
            y,
            mu_u,
            logvar_u,
            mu_z,
            logvar_z,
            mu_z_uy,
            logvar_z_uy,
            self.gammax,
            self.gammay,
        )
        loss = mse_x + kld_u + mse_y + kld_z
        logs = {
            "Loss/loss": loss.item(),
            "Loss/mse_x": mse_x.item(),
            "Loss/kld_u": kld_u.item(),
            "Loss/mse_y": mse_y.item(),
            "Loss/kld_z": kld_z.item(),
        }
        return loss, logs

    def val_step(self, batch, device):
        y, x = batch
        y, x = y.to(device), x.to(device)
        with torch.no_grad():
            x_hat, y_hat, mu_z, logvar_z, mu_u, logvar_u, mu_z_uy, logvar_z_uy = self(
                x, y
            )
            mse_x, kld_u, mse_y, kld_z = cond_loss(
                x_hat,
                x,
                y_hat,
                y,
                mu_u,
                logvar_u,
                mu_z,
                logvar_z,
                mu_z_uy,
                logvar_z_uy,
                self.gammax,
                self.gammay,
            )
            loss = mse_x + kld_u + mse_y + kld_z
        logs = {
            "Loss/val_loss": loss.item(),
            "Loss/val_mse_x": mse_x.item(),
            "Loss/val_kld_u": kld_u.item(),
            "Loss/val_mse_y": mse_y.item(),
            "Loss/val_kld_z": kld_z.item(),
        }
        return loss, logs

    def evaluate(self, val_loader, wandb_run, epoch, full_val=False):
        # CondVAE eval: aggregate SSIM & LPIPS over full validation set

        device = next(self.parameters()).device

        if full_val:
            total = {
                "ssim_y": 0.0,
                "lpips_y": 0.0,
                "ssim_x": 0.0,
                "lpips_x": 0.0,
                "ssim_sr": 0.0,
                "lpips_sr": 0.0,
            }
            count = 0
            first = True
            for batch in val_loader:
                y, x = [t.to(device) for t in batch]
                with torch.no_grad():
                    x_hat, y_hat, *_ = self.forward(x, y)
                    x_sr = self.conditional_generation(y)

                b = y.size(0)
                count += b

                for oy, ry, ox, rx, gen in zip(y, y_hat, x, x_hat, x_sr):
                    ssim_y = self.ssim(
                        oy.cpu().numpy(),
                        ry.cpu().numpy(),
                        win_size=11,
                        data_range=1.0,
                        channel_axis=0,
                    )
                    total["ssim_y"] += ssim_y
                    total["lpips_y"] += self.lpips_fn(
                        oy[[2, 1, 0]].unsqueeze(0), ry[[2, 1, 0]].unsqueeze(0)
                    ).item()
                    ssim_x = self.ssim(
                        ox.cpu().numpy(),
                        rx.cpu().numpy(),
                        win_size=11,
                        data_range=1.0,
                        channel_axis=0,
                    )
                    total["ssim_x"] += ssim_x
                    total["lpips_x"] += self.lpips_fn(
                        ox[[2, 1, 0]].unsqueeze(0), rx[[2, 1, 0]].unsqueeze(0)
                    ).item()
                    ssim_sr = self.ssim(
                        ox.cpu().numpy(),
                        gen.cpu().numpy(),
                        win_size=11,
                        data_range=1.0,
                        channel_axis=0,
                    )
                    total["ssim_sr"] += ssim_sr
                    total["lpips_sr"] += self.lpips_fn(
                        ox[[2, 1, 0]].unsqueeze(0), gen[[2, 1, 0]].unsqueeze(0)
                    ).item()

                if first:
                    imgs = {
                        "y_hat": y_hat[:4],
                        "x_hat": x_hat[:4],
                        "x_sr": x_sr[:4],
                    }
                    first = False

            # average metrics
            avg = {k: total[k] / count for k in total}

            # log aggregate metrics
            wandb_run.log(
                {
                    "Metrics/SSIM_LR": avg["ssim_y"],
                    "Metrics/LPIPS_LR": avg["lpips_y"],
                    "Metrics/SSIM_HR": avg["ssim_x"],
                    "Metrics/LPIPS_HR": avg["lpips_x"],
                    "Metrics/SSIM_SR": avg["ssim_sr"],
                    "Metrics/LPIPS_SR": avg["lpips_sr"],
                    "Metrics/SSIM_Baseline": self.ssim_base,
                    "Metrics/LPIPS_Baseline": self.lpips_base,
                },
                step=epoch,
            )
        else:
            batch = next(iter(val_loader))
            y, x = [t.to(device) for t in batch]
            with torch.no_grad():
                x_hat, y_hat, *_ = self.forward(x, y)
                x_sr = self.conditional_generation(y)

            imgs = {
                "y_hat": y_hat[:4],
                "x_hat": x_hat[:4],
                "x_sr": x_sr[:4],
            }

        # log sample images
        wandb_run.log(
            {
                "Images/LR_Recon": [
                    wandb.Image(img.permute(1, 2, 0).cpu().numpy())
                    for img in imgs["y_hat"]
                ],
                "Images/HR_Recon": [
                    wandb.Image(img.permute(1, 2, 0).cpu().numpy())
                    for img in imgs["x_hat"]
                ],
                "Images/SR_Output": [
                    wandb.Image(img.permute(1, 2, 0).cpu().numpy())
                    for img in imgs["x_sr"]
                ],
            },
            step=epoch,
        )

    def on_train_start(self, **kwargs):
        self.gammax.requires_grad = True
        self.gammay.requires_grad = True
        device = next(self.parameters()).device
        self.optimizer.add_param_group(
            {
                "params": [self.gammax, self.gammay],
            }
        )
        val_loader = self.val_loader
        if val_loader is None:
            raise ValueError(
                "Validation loader must be provided for baseline evaluation."
            )
        if os.path.exists("baseline_ckpt.pth"):
            baseline = torch.load("baseline_ckpt.pth")
            self.ssim_base = baseline["ssim_base"]
            self.lpips_base = baseline["lpips_base"]
            print(
                f"Baseline SSIM: {self.ssim_base}, LPIPS: {self.lpips_base}. Skipping baseline computation."
            )
            return
        else:
            ssim_cumu, lpips_cumu = 0, 0
            for _, batch in tqdm(enumerate(val_loader)):
                y_val, x_val = batch
                y_val, x_val = y_val.to(device), x_val.to(device)

                hr_interp = F.interpolate(y_val, scale_factor=2, mode="bicubic")

                # Compute SSIM and LPIPS scores
                for bcb, hr in zip(hr_interp, x_val):
                    ssim_val = self.ssim(
                        hr.numpy(),
                        bcb.numpy(),
                        win_size=11,
                        data_range=1.0,
                        channel_axis=0,
                    )
                    lpips = self.lpips_fn(
                        hr[[2, 1, 0]].unsqueeze(0), bcb[[2, 1, 0]].unsqueeze(0)
                    ).item()
                    ssim_cumu += ssim_val
                    lpips_cumu += lpips
            self.ssim_base = ssim_cumu / x_val.shape[0] * len(val_loader)
            self.lpips_base = lpips_cumu / x_val.shape[0] * len(val_loader)
            torch.save(
                {
                    "ssim_base": self.ssim_base,
                    "lpips_base": self.lpips_base,
                },
                "baseline_ckpt.pth",
            )
            print(
                f"Baseline SSIM: {self.ssim_base}, LPIPS: {self.lpips_base}. Baseline computation complete."
            )

    def on_train_epoch_end(self, **kwargs):
        self.wandb_run.log(
            {
                "HyperParameters/Gamma_X": self.gammax.item(),
                "HyperParameters/Gamma_Y": self.gammay.item(),
                "HyperParameters/Learning Rate": self.scheduler.get_last_lr()[0],
            },
            step=self.current_epoch,
        )


if __name__ == "__main__":
    # Example usage
    model = Cond_SRVAE(latent_size=2048, patch_size=64)
    print(model)
    x = torch.randn(1, 4, 64, 64)
    y = torch.randn(1, 4, 32, 32)  # Example condition

    x_hat, y_hat, mu_z, logvar_z, mu_u, logvar_u, mu_z_uy, logvar_z_uy = model(x, y)
    print("x_hat shape:", x_hat.shape)
    print("y_hat shape:", y_hat.shape)
    print("mu_z shape:", mu_z.shape)
    print("logvar_z shape:", logvar_z.shape)
    print("mu_u shape:", mu_u.shape)
    print("logvar_u shape:", logvar_u.shape)
    print("mu_z_uy shape:", mu_z_uy.shape)
    print("logvar_z_uy shape:", logvar_z_uy.shape)

    # You can add more code here to test the model, e.g., training loop, etc.
    # Note: This is just a placeholder for demonstration purposes.
    pass
