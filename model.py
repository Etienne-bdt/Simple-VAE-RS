import abc

import torch
import torch.nn as nn
import wandb
from tqdm import tqdm

from utils import EarlyStopper, SrEvaluator


class BaseVAE(nn.Module, metaclass=abc.ABCMeta):
    """
    Base class for all VAEs. Defines the common interface for training and validation.
    """

    def __init__(self):
        super().__init__()
        self.optimizer: torch.optim.Optimizer = torch.optim.Adam(
            self.parameters(), lr=1e-3
        )
        # Scheduler to reduce learning rate on plateau
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=30
        )
        self.latent_size: int = 0
        self.patch_size: int = 0
        self.early_stopper = EarlyStopper(patience=30, delta=0.01)

    def fit(self, train_loader, val_loader, device, epochs=1000, **kwargs):
        """
        Fit the model to the training data.
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            device: device to use for training (e.g., 'cuda' or 'cpu')
            epochs: number of epochs to train
        """
        start_epoch = kwargs.get("start_epoch", 1)
        wandb_run = wandb.init(
            project=self.__class__.__name__,
            name=f"Latent-{self.latent_size}-Patch-{self.patch_size}-SLURM-{kwargs.get('slurm_job_id', 'local')}",
            entity="ebardet-isae-supaero",
            config=kwargs.get("config", {}),
        )

        self.evaluator = SrEvaluator(val_loader, start_epoch, wandb_run=wandb_run)

        optimizer = self.optimizer
        for epoch in range(epochs):
            self.train()
            train_loss = 0.0
            for _, batch in tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                desc=f"Training, Epoch {epoch}/{epochs}",
                unit="batch",
            ):
                optimizer.zero_grad()
                loss, terms = self.train_step(batch, device)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            self.eval()

    @abc.abstractmethod
    def loss_fn(self, *args, **kwargs):
        """
        Abstract method to define the loss function.
        Args:
            args: arguments required to compute the loss
            kwargs: optional arguments
        Returns:
            loss: (torch.Tensor) computed loss value
            logs: (dict) dictionary of individual loss components
        """
        pass

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def train_step(self, batch, device, **kwargs):
        """
        Performs a training step on a batch.
        Args:
            batch: data batch
            device: device to use
        Returns:
            loss, logs (dict)
        """
        pass

    @abc.abstractmethod
    def val_step(self, batch, device, loss_fn, **kwargs):
        """
        Performs a validation step on a batch.
        Args:
            batch: data batch
            device: device to use
            loss_fn: loss function
        Returns:
            loss, logs (dict)
        """
        pass

    def log(self, wandb_run, logs, step=None):
        """
        Optional method to log model-specific information.
        Args:
            wandb_run: wandb run instance for logging
            logs: dictionary of values to log
            step: current step (optional)
        """
        if not wandb_run:
            print("WandB run not initialized, skipping logging.")
            return
        if step is not None:
            wandb_run.log(logs, step=step)


class VAE(BaseVAE):
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

    def train_step(self, batch, device, loss_fn, **kwargs):
        x = batch.to(device)
        x_hat, mu, logvar = self(x)
        loss, kld = loss_fn(
            x_hat, x, mu, logvar, kwargs.get("gamma", torch.tensor(1.0, device=device))
        )
        logs = {"loss": loss.item(), "kld": kld.item()}
        return loss, logs

    def val_step(self, batch, device, loss_fn, **kwargs):
        x = batch.to(device)
        with torch.no_grad():
            x_hat, mu, logvar = self(x)
            loss, kld = loss_fn(
                x_hat,
                x,
                mu,
                logvar,
                kwargs.get("gamma", torch.tensor(1.0, device=device)),
            )
        logs = {"val_loss": loss.item(), "val_kld": kld.item()}
        return loss, logs


class Cond_SRVAE(BaseVAE):
    def __init__(self, latent_size, patch_size=256):
        super(Cond_SRVAE, self).__init__()
        self.latent_size = latent_size
        self.latent_size_y = latent_size // 4
        self.patch_size = patch_size
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

    def train_step(self, batch, device, loss_fn, **kwargs):
        y, x = batch
        y, x = y.to(device), x.to(device)
        x_hat, y_hat, mu_z, logvar_z, mu_u, logvar_u, mu_z_uy, logvar_z_uy = self(x, y)
        mse_x, kld_u, mse_y, kld_z = loss_fn(
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
            kwargs.get("gamma", torch.tensor(1.0, device=device)),
            kwargs.get("gamma2", torch.tensor(1.0, device=device)),
        )
        loss = (
            mse_x + kld_u + mse_y + kld_z
            if not kwargs.get("pretrain", False)
            else mse_y + kld_u + mse_x
        )
        logs = {
            "loss": loss.item(),
            "mse_x": mse_x.item(),
            "kld_u": kld_u.item(),
            "mse_y": mse_y.item(),
            "kld_z": kld_z.item(),
        }
        return loss, logs

    def val_step(self, batch, device, loss_fn, **kwargs):
        y, x = batch
        y, x = y.to(device), x.to(device)
        with torch.no_grad():
            x_hat, y_hat, mu_z, logvar_z, mu_u, logvar_u, mu_z_uy, logvar_z_uy = self(
                x, y
            )
            mse_x, kld_u, mse_y, kld_z = loss_fn(
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
                kwargs.get("gamma", torch.tensor(1.0, device=device)),
                kwargs.get("gamma2", torch.tensor(1.0, device=device)),
            )
            loss = (
                mse_x + kld_u + mse_y + kld_z
                if not kwargs.get("pretrain", False)
                else mse_y + kld_u + mse_x
            )
        logs = {
            "val_loss": loss.item(),
            "val_mse_x": mse_x.item(),
            "val_kld_u": kld_u.item(),
            "val_mse_y": mse_y.item(),
            "val_kld_z": kld_z.item(),
        }
        return loss, logs


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
    print("All size tests passed!")
