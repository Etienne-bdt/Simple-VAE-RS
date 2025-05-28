import os

import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
import wandb

from loss import cond_loss

from .base import BaseVAE


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

        self.encoder1 = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1),  # 4 input channels (
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 32 input channels
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 64 input channels
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.fc_mu_1 = nn.Linear(patch_size**2 // 2, self.latent_size_y)
        self.fc_logvar_1 = nn.Sequential(
            nn.Linear(patch_size**2 // 2, self.latent_size_y),
            nn.Hardtanh(-7, 7),
        )
        self.fc_decode_y = nn.Linear(self.latent_size_y, patch_size**2 // 2)
        self.decoder_1 = nn.Sequential(
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

        self.encoder2 = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1),  # 4 input channels (
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 32 input channels
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 64 input channels
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.fc_mu_2 = nn.Linear(2 * patch_size**2, self.latent_size)
        self.fc_logvar_2 = nn.Sequential(
            nn.Linear(2 * patch_size**2, self.latent_size),
            nn.Hardtanh(-7, 7),
        )
        self.fc_decode_x = nn.Linear(self.latent_size * 2, 2 * patch_size**2)
        self.decoder_2 = nn.Sequential(
            nn.ConvTranspose2d(
                128, 64, kernel_size=4, stride=2, padding=1
            ),  # 128 input channels
            nn.ReLU(),
            nn.ConvTranspose2d(
                64, 32, kernel_size=4, stride=2, padding=1
            ),  # 64 input channels
            nn.ReLU(),
            nn.ConvTranspose2d(
                32, 32, kernel_size=3, stride=1, padding=1
            ),  # 32 input channels
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 4, kernel_size=2, stride=1, padding=0),
            nn.Sigmoid(),
        )
        """
        self.decoder_hf = nn.Sequential(
            nn.ConvTranspose2d(36, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 4, kernel_size=2, stride=1, padding=0),
            nn.Sigmoid(),
        )
        
        self.decoder_hf = nn.Sequential(
            nn.ConvTranspose2d(36, 32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 4, kernel_size=2, stride=1, padding=0),
            nn.Sigmoid(),
        )
        """
        self.y_to_z = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1),  # 4 input channels (
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 32 input channels
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 64 input channels
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(patch_size**2 // 2, self.latent_size),
        )
        self.u_to_z = nn.Linear(self.latent_size_y, self.latent_size)
        self.mu_u_y_to_z = nn.Linear(self.latent_size * 2, self.latent_size)
        self.logvar_u_y_to_z = nn.Sequential(
            nn.Linear(self.latent_size * 2, self.latent_size),
            nn.Hardtanh(-7, 7),
        )

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
        y = self.encoder1(y)
        y = y.view(y.size(0), -1)
        mu_y = self.fc_mu_1(y)
        logvar_y = self.fc_logvar_1(y)
        return mu_y, logvar_y

    def encode_x(self, x):
        # Define the encoder part of the VAE
        x = self.encoder2(x)
        x = x.view(x.size(0), -1)
        mu_x = self.fc_mu_2(x)
        logvar_x = self.fc_logvar_2(x)
        return mu_x, logvar_x

    def reparameterize(self, mu, logvar):
        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode_y(self, u):
        u = self.fc_decode_y(u)
        u = u.view(u.size(0), 128, self.patch_size // 16, self.patch_size // 16)
        y = self.decoder_1(u)
        return y

    def decode_x(self, z, y):
        y_enc = self.y_to_z(y)
        stack = torch.cat((y_enc, z), dim=1)
        z = self.fc_decode_x(stack)
        z = z.view(y.size(0), 128, self.patch_size // 8, self.patch_size // 8)
        return self.decoder_2(z)

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
            "loss": loss.item(),
            "mse_x": mse_x.item(),
            "kld_u": kld_u.item(),
            "mse_y": mse_y.item(),
            "kld_z": kld_z.item(),
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
            "val_loss": loss.item(),
            "val_mse_x": mse_x.item(),
            "val_kld_u": kld_u.item(),
            "val_mse_y": mse_y.item(),
            "val_kld_z": kld_z.item(),
        }
        return loss, logs

    def evaluate(self, val_loader, wandb_run, epoch):
        # CondVAE eval: aggregate SSIM & LPIPS over full validation set

        device = next(self.parameters()).device

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
                    "y": y[:4],
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
                "CondVAE/SSIM_LR": avg["ssim_y"],
                "CondVAE/LPIPS_LR": avg["lpips_y"],
                "CondVAE/SSIM_HR": avg["ssim_x"],
                "CondVAE/LPIPS_HR": avg["lpips_x"],
                "CondVAE/SSIM_SR": avg["ssim_sr"],
                "CondVAE/LPIPS_SR": avg["lpips_sr"],
                "CondVAE/SSIM_Baseline": self.ssim_base,
                "CondVAE/LPIPS_Baseline": self.lpips_base,
            },
            step=epoch,
        )

        # log sample images
        wandb_run.log(
            {
                "CondVAE/LR_Recon": [
                    wandb.Image(img.permute(1, 2, 0).cpu().numpy())
                    for img in imgs["y_hat"]
                ],
                "CondVAE/HR_Recon": [
                    wandb.Image(img.permute(1, 2, 0).cpu().numpy())
                    for img in imgs["x_hat"]
                ],
                "CondVAE/SR_Output": [
                    wandb.Image(img.permute(1, 2, 0).cpu().numpy())
                    for img in imgs["x_sr"]
                ],
            },
            step=epoch,
        )

    def on_train_start(self, **kwargs):
        self.gammax.requires_grad = True
        self.gammay.requires_grad = True
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
                y_val = y_val.cuda()
                x_val = x_val.cuda()

                hr_interp = F.interpolate(y_val, scale_factor=2, mode="bicubic")

                # Compute SSIM and LPIPS scores
                for bcb, hr in zip(hr_interp, x_val):
                    ssim_val = self.ssim(
                        hr, bcb, win_size=11, data_range=1.0, channel_axis=0
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
                "CondVAE/Gamma_X": self.gammax.item(),
                "CondVAE/Gamma_Y": self.gammay.item(),
                "CondVAE/Learning Rate": self.scheduler.get_last_lr()[0],
            },
            step=self.current_epoch,
        )
