import lightning as L
import torch
import torch.nn as nn

from loss import cond_loss, loss_function


class VAE(nn.Module):
    def __init__(self, latent_size):
        super(VAE, self).__init__()
        self.latent_size = latent_size
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
        self._initialize_weights()  # Add weight initialization

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

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


class VAE_Lightning(L.LightningModule):
    def __init__(self, latent_size):
        super(VAE_Lightning, self).__init__()
        self.model = VAE(latent_size)
        self.gamma = torch.nn.Parameter(torch.ones(1), requires_grad=True)

    def training_step(self, batch, batch_idx):
        _, data = batch
        recon_batch, mu, logvar = self.model(data)
        mse, kld = loss_function(recon_batch, data, mu, logvar, self.gamma)
        loss = mse + kld
        values = {"loss": loss, "mse": mse, "kld": kld}
        self.log_dict(
            values, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True
        )
        self.log(
            "grad_norm", self._compute_grad_norm(), prog_bar=True
        )  # Log gradient norm
        return loss

    def _compute_grad_norm(self):
        total_norm = 0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm**0.5

    def validation_step(self, batch, batch_idx):
        _, data = batch
        recon_batch, mu, logvar = self.model(data)
        mse, kld = loss_function(recon_batch, data, mu, logvar, self.gamma)
        loss = mse + kld
        values = {
            "val_loss": loss,
            "val_mse": mse,
            "val_kld": kld,
            "gamma": self.gamma.item(),
        }
        self.log_dict(values, on_epoch=True, on_step=False, sync_dist=True)
        return loss

    def on_train_end(self) -> None:
        print(f"Final gamma value: {self.gamma.item()}")
        return super().on_train_end()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        optimizer.add_param_group({"params": self.gamma})
        return optimizer


class Cond_SRVAE(nn.Module):
    def __init__(self, latent_size, patch_size=256):
        super(Cond_SRVAE, self).__init__()
        self.latent_size = latent_size
        self.patch_size = patch_size
        self.encoder1 = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1),  # 4 input channels (
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 32 input channels
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 64 input channels
            nn.ReLU(),
        )
        self.fc_mu_1 = nn.Linear(patch_size**2 // 2, self.latent_size)
        self.fc_logvar_1 = nn.Linear(patch_size**2 // 2, self.latent_size)
        self.fc_decode_y = nn.Linear(self.latent_size, patch_size**2 // 2)
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
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 32 input channels
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 64 input channels
            nn.ReLU(),
        )
        self.fc_mu_2 = nn.Linear(2 * patch_size**2, self.latent_size)
        self.fc_logvar_2 = nn.Linear(2 * patch_size**2, self.latent_size)
        self.fc_decode_x = nn.Linear(self.latent_size, 2 * patch_size**2)
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
                32, 4, kernel_size=3, stride=1, padding=1
            ),  # 32 input channels
        )
        self.decoder_hf = nn.Sequential(
            nn.ConvTranspose2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 4, kernel_size=2, stride=1, padding=0),
            nn.Sigmoid(),
        )

        self.y_to_z = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1),  # 4 input channels (
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 32 input channels
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 64 input channels
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(patch_size**2 // 2, self.latent_size // 2),
        )
        self.u_to_z = nn.Linear(self.latent_size, self.latent_size // 2)
        self.mu_u_y_to_z = nn.Linear(self.latent_size, self.latent_size)
        self.logvar_u_y_to_z = nn.Linear(self.latent_size, self.latent_size)

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
        z = self.fc_decode_x(z)
        z = z.view(z.size(0), 128, self.patch_size // 8, self.patch_size // 8)
        z_carac = self.decoder_2(z)
        stack = torch.cat((y, z_carac), dim=1)
        return self.decoder_hf(stack)

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

    def sample(self, y, samples=1000):
        # Generate samples from the model
        mu_u, logvar_u = self.encode_y(y)
        u = self.reparameterize(mu_u, logvar_u)

        mu_z_uy, logvar_z_uy = self.z_cond(y, u)

        std = torch.exp(0.5 * logvar_z_uy)
        latent = std.size(1)
        eps = torch.randn((samples, latent)).to(std.device)

        z = mu_z_uy + eps * std

        return self.decode_x(z, y)

    def freeze_cond(self):
        for param in self.u_to_z.parameters():
            param.requires_grad = False
        for param in self.mu_u_y_to_z.parameters():
            param.requires_grad = False
        for param in self.logvar_u_y_to_z.parameters():
            param.requires_grad = False
        for param in self.y_to_z.parameters():
            param.requires_grad = False

    def unfreeze_cond(self):
        for param in self.u_to_z.parameters():
            param.requires_grad = True
        for param in self.mu_u_y_to_z.parameters():
            param.requires_grad = True
        for param in self.logvar_u_y_to_z.parameters():
            param.requires_grad = True
        for param in self.y_to_z.parameters():
            param.requires_grad = True


class Cond_SRVAE_Lightning(L.LightningModule):
    def __init__(self, latent_size):
        super(Cond_SRVAE_Lightning, self).__init__()
        self.model = Cond_SRVAE(latent_size)
        self.gamma = torch.nn.Parameter(torch.ones(1), requires_grad=True)

    def training_step(self, batch, batch_idx):
        y, x = batch
        x_hat, y_hat, mu_z, logvar_z, mu_u, logvar_u, mu_z_uy, logvar_z_uy = self.model(
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
            self.gamma,
        )
        loss = mse_x + kld_u + mse_y + kld_z
        values = {
            "loss": loss,
            "mse_x": mse_x,
            "kld_u": kld_u,
            "mse_y": mse_y,
            "kld_z": kld_z,
            "logvar_z": logvar_z.mean(),
            "logvar_z_uy": logvar_z_uy.mean(),
            "mu_z_uy": mu_z_uy.mean(),
            "mu_z": mu_z.mean(),
        }
        self.log_dict(
            values, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        y, x = batch
        x_hat, y_hat, mu_z, logvar_z, mu_u, logvar_u, mu_z_uy, logvar_z_uy = self.model(
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
            self.gamma,
        )
        loss = mse_x + kld_u + mse_y + kld_z
        values = {
            "val_loss": loss,
            "val_mse_x": mse_x,
            "val_kld_u": kld_u,
            "val_mse_y": mse_y,
            "val_kld_z": kld_z,
            "gamma": self.gamma.item(),
        }
        self.log_dict(values, on_epoch=True, on_step=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-4)
        optimizer.add_param_group({"params": self.gamma})
        return optimizer


if __name__ == "__main__":
    LATENT_SIZE = 128
    PATCH_SIZE = 64
    print("Testing model size")
    model = Cond_SRVAE(LATENT_SIZE, PATCH_SIZE)
    x = torch.randn(1, 4, PATCH_SIZE, PATCH_SIZE)
    y = torch.randn(1, 4, PATCH_SIZE // 2, PATCH_SIZE // 2)
    x_hat, y_hat, mu_z, logvar_z, mu_u, logvar_u, mu_z_uy, logvar_z_uy = model(x, y)

    print(x_hat.shape)
    print(y_hat.shape)

    assert x_hat.shape == x.shape
    assert y_hat.shape == y.shape
    for i, var in enumerate([mu_z, logvar_z, mu_u, logvar_u, mu_z_uy, logvar_z_uy]):
        print(f"Testing {i}")
        assert var.shape == (1, LATENT_SIZE)
    print("All size tests passed!")
