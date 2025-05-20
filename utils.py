import os

import lpips
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm


def normalize_image(image: torch.Tensor) -> torch.Tensor:
    """
    Normalize the image tensor to the range [0, 1] for visualization.
    Args:
        image (torch.Tensor): Input image tensor.
    Returns:
        torch.Tensor: Normalized image tensor.
    """
    if image.ndim == 3:
        # If the image is 3D (C, H, W), add a batch dimension
        min_val = image.amin(dim=(1, 2), keepdim=True)
        max_val = image.amax(dim=(1, 2), keepdim=True)
        normalized_image = (image - min_val) / (max_val - min_val + 1e-5)
    elif image.ndim == 4:
        min_val = image.amin(dim=(2, 3), keepdim=True)
        max_val = image.amax(dim=(2, 3), keepdim=True)
        normalized_image = (image - min_val) / (max_val - min_val + 1e-5)
    else:
        raise ValueError("Input image must be 3D or 4D tensor.")
    return normalized_image


class EarlyStopper:
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

    def __call__(self, val_loss: float) -> bool:
        """
        Call this method to check if training should be stopped.
        Args:
            val_loss (float): Current validation loss.
        """
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        elif val_loss > self.best_loss + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class SrEvaluator:
    """
    Class to evaluate the performance of the super-resolution (SR) model.
    Args:
        y_val (torch.Tensor): Low-resolution images.
        x_val (torch.Tensor): High-resolution images.
        writer (SummaryWriter): TensorBoard writer.
        start_epoch (int): Current epoch number.
    """

    def __init__(self, val_loader, start_epoch):
        """
        Initialize the SR_Evaluator class.
        """
        self.val_loader = val_loader
        self.len_val = len(val_loader)
        self.writer = SummaryWriter()
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
        self.writer.add_images(
            "Conditional Generation/HR Interpolated",
            hr_interp[:, [2, 1, 0], :, :],
            global_step=self.start_epoch,
            dataformats="NCHW",
        )

        self.writer.add_images(
            "Conditional Generation/HR_Original",
            x_val[:4, [2, 1, 0], :, :],
            global_step=self.start_epoch,
            dataformats="NCHW",
        )
        self.writer.add_images(
            "Conditional Generation/LR_Original",
            y_val[:4, [2, 1, 0], :, :],
            global_step=self.start_epoch,
            dataformats="NCHW",
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
        ssim_score = ssim_score
        ssim_score = torch.tensor(ssim_score) / (b * self.len_val)

        lpips_score = lpips_score / (b * self.len_val)
        return ssim_score, lpips_score

    def log_images(self, img, category, epoch):
        """
        Log the images to TensorBoard.
        Args:
            img (torch.Tensor): Image tensor to log.
            category (str): Category of the image.
            epoch (int): Current epoch number.
        """
        self.writer.add_images(
            category,
            img[:4, [2, 1, 0], :, :],
            global_step=epoch,
            dataformats="NCHW",
        )
