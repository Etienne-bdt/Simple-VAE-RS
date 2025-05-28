import abc
from typing import List

import lpips
import torch
import torch.nn as nn
from skimage import metrics as skmetrics
from tqdm import tqdm
import wandb

from callbacks import Callback


class BaseVAE(nn.Module, metaclass=abc.ABCMeta):
    """
    Base class for all VAEs. Defines the common interface for training and validation.
    """

    def __init__(self, patch_size: int = 64, callbacks: List[Callback] | None = None):
        if callbacks is None:
            callbacks = []
        super().__init__()
        # Scheduler to reduce learning rate on plateau
        self.latent_size: int = 0
        self.patch_size: int = patch_size
        self.callbacks: List[Callback] = callbacks
        self.ssim = skmetrics.structural_similarity
        self.lpips_fn = lpips.LPIPS(net="alex")

    def fit(self, train_loader, val_loader, device, optimizer, epochs=1000, **kwargs):
        """
        Fit the model to the training data.
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            device: device to use for training (e.g., 'cuda' or 'cpu')
            epochs: number of epochs to train
        """
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=10
        )
        self.current_epoch: int = 0
        self.lpips_fn = self.lpips_fn.to(device)
        self.lpips_fn.eval()
        start_epoch = kwargs.get("start_epoch", 1)
        val_metrics_every = kwargs.get("val_metrics_every", float("inf"))
        x, _ = next(iter(train_loader))
        b = x.size(0)

        self.wandb_run = wandb.init(
            project=self.__class__.__name__,
            name=f"Latent-{self.latent_size}-Patch-{self.patch_size}-SLURM-{kwargs.get('slurm_job_id', 'local')}",
            entity="ebardet-isae-supaero",
            config=kwargs.get(
                "config",
                {
                    "latent_size": self.latent_size,
                    "patch_size": self.patch_size,
                    "epochs": epochs,
                    "batch_size": b,
                    "val_metrics_every": val_metrics_every,
                    "slurm_job_id": kwargs.get("slurm_job_id", "local"),
                },
            ),
        )

        optimizer = self.optimizer
        self.on_train_start()

        for epoch in range(start_epoch, epochs + 1):
            self.current_epoch = epoch
            for cb in self.callbacks:
                if cb.on_epoch_begin(
                    epoch=epoch, optimizer=optimizer, device=device, model=self
                ):
                    print(
                        f"Stopping training before epoch {epoch} due to {cb.__class__.__name__} condition."
                    )
                    return  # Stop training if callback indicates to stop
            self.train()
            train_loss = 0.0
            terms_dict = {}
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
                if not terms_dict:
                    terms_dict = terms
                else:
                    for key, value in terms.items():
                        if key in terms_dict:
                            terms_dict[key] += value
                        else:
                            terms_dict[key] = value
                train_loss += loss.item()

            # Average the loss terms
            for key in terms_dict:
                terms_dict[key] /= len(train_loader)
            self.terms_dict = terms_dict
            train_loss /= len(train_loader)

            self.on_train_epoch_end()
            self.eval()
            val_loss = 0.0
            val_terms_dict = {}
            with torch.no_grad():
                for _, batch in tqdm(
                    enumerate(val_loader),
                    total=len(val_loader),
                    desc=f"Validation, Epoch {epoch}/{epochs}",
                    unit="batch",
                ):
                    loss, terms = self.val_step(batch, device)
                    if not val_terms_dict:
                        val_terms_dict = terms
                    else:
                        for key, value in terms.items():
                            if key in val_terms_dict:
                                val_terms_dict[key] += value
                            else:
                                val_terms_dict[key] = value

                    val_loss += loss.item()

                if epoch % val_metrics_every == 0 or epoch in [1, epochs]:
                    self.evaluate(val_loader, self.wandb_run, epoch)

            # Average the validation loss terms
            for key in val_terms_dict:
                val_terms_dict[key] /= len(val_loader)

            val_loss /= len(val_loader)
            if self.scheduler:
                self.scheduler.step(val_loss)
            self.log(self.wandb_run, val_terms_dict, step=epoch)
            for cb in self.callbacks:
                if cb.on_epoch_end(
                    epoch=epoch,
                    optimizer=optimizer,
                    device=device,
                    model=self,
                    logs=val_terms_dict,
                ):
                    print(
                        f"Stopping training after epoch {epoch} due to {cb.__class__.__name__} condition."
                    )
                    return  # Stop training if callback indicates to stop

            print(
                f"Epoch {epoch}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

        self.wandb_run.finish()

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        """
        Abstract method for the forward pass of the model.
        """
        raise NotImplementedError("forward must be implemented in the derived class.")

    @abc.abstractmethod
    def train_step(self, batch, device):
        """
        Performs a training step on a batch.
        Args:
            batch: data batch
            device: device to use
        Returns:
            loss: scalar loss value
            logs (dict): dictionary of values to log, e.g., {'loss': loss_value, 'kld': kld_value}
        """
        raise NotImplementedError(
            "train_step must be implemented in the derived class."
        )

    @abc.abstractmethod
    def val_step(self, batch, device):
        """
        Performs a validation step on a batch.
        Args:
            batch: data batch
            device: device to use
            loss_fn: loss function
        Returns:
            loss: scalar loss value
            logs (dict): dictionary of values to log, e.g., {'val_loss': loss_value, 'val_kld': kld_value}
        """
        raise NotImplementedError("val_step must be implemented in the derived class.")

    @abc.abstractmethod
    def evaluate(self, val_loader, wandb_run, epoch):
        """
        Evaluate the model on the validation set.
        Args:
            val_loader: DataLoader for validation data
            wandb_run: wandb run instance for logging
            epoch: current epoch number
        """
        raise NotImplementedError("evaluate must be implemented in the derived class.")

    def log(self, wandb_run, logs: dict, step=None):
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

    @abc.abstractmethod
    def on_train_start(self, **kwargs):
        """
        Called at the start of training.
        Args:
            kwargs: additional arguments
        """
        pass

    @abc.abstractmethod
    def on_train_epoch_end(self, **kwargs):
        """
        Called at the end of each training epoch.
        Args:
            kwargs: additional arguments
        """
        pass
