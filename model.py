import torch
import torch.nn as nn
import lightning as L
from loss import loss_function

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
            nn.ReLU())
        self.fc_mu = nn.Linear(128 * 8 * 8 *16, self.latent_size)
        self.fc_logvar = nn.Linear(128 * 8 * 8 * 16, self.latent_size)
        self.fc_decode = nn.Linear(self.latent_size, 128 * 8 * 8*16)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1),  # 128 input channels
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 64 input channels
            nn.ReLU(),
            nn.ConvTranspose2d(32, 4, kernel_size=8, stride=2, padding=1),  # 32 input channels
            nn.Sigmoid())
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
        z = z.view(z.size(0), 128, 8*4, 8*4)
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
        values = { "loss": loss, "mse": mse, "kld": kld}
        self.log_dict(values, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss
    

    def validation_step(self, batch, batch_idx):
        _, data = batch
        recon_batch, mu, logvar = self.model(data)
        mse, kld = loss_function(recon_batch, data, mu, logvar, self.gamma)
        loss = mse + kld
        values = { "val_loss": loss, "val_mse": mse, "val_kld": kld , "gamma": self.gamma.item()}
        self.log_dict(values, on_epoch=True, on_step=False, sync_dist=True)
        return loss

    def on_train_end(self) -> None:
        print(f"Final gamma value: {self.gamma.item()}")
        return super().on_train_end()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        optimizer.add_param_group({'params': self.gamma})
        return optimizer
    