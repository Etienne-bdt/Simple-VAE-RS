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