import torch

def normalize_image(image: torch.Tensor) -> torch.Tensor:
    """
    Normalize the image tensor to the range [0, 1] for visualization.
    Args:
        image (torch.Tensor): Input image tensor.
    Returns:
        torch.Tensor: Normalized image tensor.
    """
    min_val = image.min()
    max_val = image.max()
    normalized_image = (image - min_val) / (max_val - min_val + 1e-5)
    return normalized_image