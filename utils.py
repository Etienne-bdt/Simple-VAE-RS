import torch


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
