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
        self.writer = SummaryWriter()
        self.start_epoch = start_epoch
        self.ssim = ssim
        self.compute_baseline()

    def compute_baseline(self):
        """
        Compute and log the baseline images for the SR task.
        Args:
            y_val (torch.Tensor): Low-resolution images.
            x_val (torch.Tensor): High-resolution images.
            writer (SummaryWriter): TensorBoard writer.
            start_epoch (int): Current epoch number.
        """
        y_val, x_val = next(reversed(list(self.val_loader)))
        hr_interp = F.interpolate(y_val[:4], scale_factor=2, mode="bicubic")
        self.writer.add_images(
            "Conditional Generation/HR Interpolated",
            hr_interp,
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
        ssim_cumu = 0
        for _, batch in tqdm(enumerate(self.val_loader)):
            y_val, x_val = batch
            y_val = y_val.cuda()
            x_val = x_val.cuda()

            hr_interp = F.interpolate(y_val, scale_factor=2, mode="bicubic")

            # Compute SSIM and LPIPS scores
            ssim = self.compute_metrics(hr_interp, x_val)
            ssim_cumu += ssim
        self.ssim_val = ssim_cumu / len(self.val_loader)

    def compute_metrics(self, pred, gt):
        """
        Compute and log the metrics for the SR task.
        Args:
            y_val (torch.Tensor): Low-resolution images.
            x_val (torch.Tensor): High-resolution images.
            writer (SummaryWriter): TensorBoard writer.
            start_epoch (int): Current epoch number.
        """
        ssim_score = self.ssim(
            pred.permute(0, 2, 3, 1).cpu().numpy(),
            gt.permute(0, 2, 3, 1).cpu().numpy(),
            multichannel=True,
        )
        ssim_score = torch.tensor(ssim_score).mean()
        return ssim_score

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
