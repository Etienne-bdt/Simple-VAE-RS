import abc
import os

import lpips
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm


class Callback(abc.ABC):
    """
    Abstract Class for defining callbacks in a training loop.
    """

    @abc.abstractmethod
    def on_epoch_begin(self, **kwargs):
        """
        Called at the beginning of each epoch.
        Returns: bool: True if training should stop, False otherwise.
        """
        return False

    @abc.abstractmethod
    def on_epoch_end(self, **kwargs):
        """
        Called at the end of each epoch.
        Returns: bool: True if training should stop, False otherwise.
        """
        return False


class EarlyStopping(Callback):
    """
    Early stopping utility to stop training when validation loss does not improve.
    Args:
        patience (int): Number of epochs with no improvement after which training will be stopped.
        delta (float): Minimum change in the monitored quantity to qualify as an improvement.
    """

    def __init__(self, patience: int = 10, delta: float = 0) -> None:
        """
        Early stopping utility to stop training when validation loss does not improve.
        Args:
            patience (int): Number of epochs with no improvement after which training will be stopped.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = float("inf")
        self.best_epoch = 0
        self.metric_name = "val_loss"

    def on_epoch_end(self, **kwargs) -> bool:
        """
        Call this method to check if training should be stopped.
        Args:
            val_loss (float): Current validation loss.
        """
        logs = kwargs.get("logs", {})
        val_loss = logs.get(self.metric_name, float("inf"))
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        elif val_loss > self.best_loss + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class ModelCheckpoint(Callback):
    """
    Class to save the model at the end of each epoch.
    """

    def __init__(
        self,
        save_path: str,
        monitor: str = "val_loss",
        mode: str = "min",
        save_best_only: bool = True,
    ) -> None:
        """
        Args:
            save_path (str): Path to save the model.
            monitor (str): Metric to monitor for saving the model.
            mode (str): One of {'min', 'max'}. In 'min' mode, the model is saved when the monitored metric decreases.
            save_best_only (bool): If True, only saves the model when the monitored metric improves.
        """
        self.save_path = save_path
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.best_metric = float("inf") if mode == "min" else float("-inf")
        self.best_epoch = 0

    def on_epoch_end(self, **kwargs) -> bool:
        """
        Called at the end of each epoch to save the model if the monitored metric improves.
        Args:
            logs (dict): Dictionary containing the metrics for the epoch.
        """
        logs = kwargs.get("logs", {})
        model = kwargs.get("model", nn.Sequential())
        current_metric = logs.get(self.monitor, float("inf"))
        if self.save_best_only:
            if (self.mode == "min" and current_metric < self.best_metric) or (
                self.mode == "max" and current_metric > self.best_metric
            ):
                self.best_metric = current_metric
                self.best_epoch = kwargs.get("epoch", 0)
                # Save the model here
                torch.save(model.state_dict(), self.save_path)
        else:
            # Save the model every epoch
            torch.save(
                model.state_dict(),
                f"{self.save_path}_epoch_{kwargs.get('epoch', 0)}.pth",
            )
        return False


class SrEvaluator:
    """
    Class to evaluate the performance of the super-resolution (SR) model.
    Args:
        y_val (torch.Tensor): Low-resolution images.
        x_val (torch.Tensor): High-resolution images.
        wandb_run: WandB run.
        start_epoch (int): Current epoch number.
    """

    def __init__(self, val_loader, start_epoch, wandb_run):
        """
        Initialize the SR_Evaluator class.
        """
        self.val_loader = val_loader
        self.len_val = len(val_loader)
        self.wandb_run = wandb_run
        self.start_epoch = start_epoch
        self.lpips_loss = lpips.LPIPS(net="alex").cuda()
        self.ssim = ssim
        self.compute_baseline()

    def compute_baseline(self):
        """
        Compute and log the baseline images for the SR task.
        """
        for _batch in self.val_loader:
            pass
        y_val, x_val = _batch
        hr_interp = F.interpolate(y_val[:4, :, :, :], scale_factor=2, mode="bicubic")
        self.wandb_run.log(
            {
                "Conditional Generation/HR Interpolated": [
                    wandb.Image(
                        hr_interp[i].permute(1, 2, 0).cpu().numpy(),
                        caption=f"HR Interpolated {i}",
                    )
                    for i in range(hr_interp.shape[0])
                ],
                "Conditional Generation/HR_Original": [
                    wandb.Image(
                        x_val[i].permute(1, 2, 0).cpu().numpy(),
                        caption=f"HR Original {i}",
                    )
                    for i in range(min(y_val.shape[0], 4))
                ],
                "Conditional Generation/LR_Original": [
                    wandb.Image(
                        y_val[i].permute(1, 2, 0).cpu().numpy(),
                        caption=f"LR Original {i}",
                    )
                    for i in range(min(y_val.shape[0], 4))
                ],
            },
            step=self.start_epoch - 1,
        )
        ssim_cumu, lpips_cumu = 0, 0
        if not os.path.exists("baseline_ckpt.pth"):
            for _, batch in tqdm(enumerate(self.val_loader)):
                y_val, x_val = batch
                y_val = y_val.cuda()
                x_val = x_val.cuda()

                hr_interp = F.interpolate(y_val, scale_factor=2, mode="bicubic")

                # Compute SSIM and LPIPS scores
                ssim, lpips = self.compute_metrics(hr_interp, x_val)
                ssim_cumu += ssim
                lpips_cumu += lpips
            self.ssim_base = ssim_cumu
            self.lpips_base = lpips_cumu
            torch.save(
                {
                    "ssim_base": self.ssim_base,
                    "lpips_base": self.lpips_base,
                },
                "baseline_ckpt.pth",
            )
        else:
            checkpoint = torch.load("baseline_ckpt.pth")
            self.ssim_base = checkpoint["ssim_base"]
            self.lpips_base = checkpoint["lpips_base"]
            print(f"SSIM Baseline: {self.ssim_base}, LPIPS Baseline: {self.lpips_base}")

    def compute_metrics(self, pred, gt):
        """
        Compute and log the metrics for the SR task.
        Args:
            pred (torch.Tensor): Predicted images (A batch).
            gt (torch.Tensor): Ground truth images (A batch).
        """
        ssim_score, lpips_score = 0, 0
        b = pred.shape[0]
        for p, g in zip(pred, gt):
            ssim_s = self.ssim(
                p.cpu().numpy(),
                g.cpu().numpy(),
                win_size=11,
                data_range=1,
                channel_axis=0,
                gradient=False,
                full=False,
            )
            lpips_s = self.lpips_loss(
                p[[2, 1, 0], :, :].cuda(),
                g[[2, 1, 0], :, :].cuda(),
            )
            lpips_score += lpips_s
            ssim_score += ssim_s
        ssim_score = torch.tensor(ssim_score) / (b * self.len_val)

        lpips_score = lpips_score / (b * self.len_val)
        return ssim_score, lpips_score

    def log_images(self, img, category, epoch):
        """
        Log the images to WandB.
        Args:
            img (torch.Tensor): Image tensor to log.
            category (str): Category of the image.
            epoch (int): Current epoch number.
        """
        self.wandb_run.log(
            {
                category: [
                    wandb.Image(
                        img[i].permute(1, 2, 0).cpu().detach().numpy(),
                        caption=f"{category} {i}",
                    )
                    for i in range(img.shape[0])
                ]
            },
            step=epoch - 1,
        )
