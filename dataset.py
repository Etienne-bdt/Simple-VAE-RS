import os

import numpy as np
import polars as pl
import rasterio
import torch
from torch.utils.data import Dataset


def init_dataloader(dataset: str, batch_size: int = 16, patch_size: int = 256):
    """
    Initialize the dataloader for the dataset.
    Args:
        dataset (str): The name of the dataset to load.
        batch_size (int): The batch size for the dataloader.
    Returns:
        train_loader (DataLoader): The dataloader for the training set.
        val_loader (DataLoader): The dataloader for the validation set.
    """
    if dataset == "Sen2Venus" or dataset == "sen2venus" or dataset == "s2v":
        ds = Sen2VenDataset(patch_size)

    elif dataset == "Floods" or dataset == "floods":
        ds = FloodDataset(patch_size=256)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    train_size = int(0.8 * len(ds))
    val_size = len(ds) - train_size
    train_ds, val_ds = torch.utils.data.random_split(ds, [train_size, val_size])
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size, shuffle=True, num_workers=6, persistent_workers=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size, shuffle=False, num_workers=6, persistent_workers=True
    )
    return train_loader, val_loader


class FloodDataset(Dataset):
    def __init__(self, patch_size=64):
        super(FloodDataset, self).__init__()
        self.patch_size = patch_size
        self.patches = []
        self.precompute_patches()

    def precompute_patches(self):
        root = os.listdir("/scratch/disc/e.bardet/Simple-VAE-RS/floods")
        for path in root:
            img_paths = [
                os.path.join(
                    "/scratch/disc/e.bardet/Simple-VAE-RS/floods", path, "S2", x
                )
                for x in os.listdir(
                    os.path.join(
                        "/scratch/disc/e.bardet/Simple-VAE-RS/floods", path, "S2"
                    )
                )
                if x.endswith(".tif")
            ]
            for img_path in img_paths:
                with rasterio.open(img_path) as src:
                    img = src.read()  # Read all bands
                height, width = img.shape[1], img.shape[2]
                for row in range(0, height, self.patch_size):
                    for col in range(0, width, self.patch_size):
                        if (
                            row + self.patch_size <= height
                            and col + self.patch_size <= width
                        ):
                            patch = img[
                                :,
                                row : row + self.patch_size,
                                col : col + self.patch_size,
                            ]
                            quantiles = np.quantile(
                                patch, [0.01, 0.99], axis=(1, 2), keepdims=True
                            )
                            patch = (patch - quantiles[0]) / (
                                quantiles[1] - quantiles[0] + 1e-5
                            )
                            patch = np.clip(patch, 0, 1)
                            patch = torch.tensor(patch, dtype=torch.float32)
                            if not torch.isnan(patch).any():
                                self.patches.append(patch)

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        return self.patches[idx]


class Sen2VenDataset(Dataset):
    def __init__(self, patch_size=256, dataset="ARM", bands="visu"):
        super(Sen2VenDataset, self).__init__()
        self.dataset = os.path.join(os.getcwd(), dataset)
        csv_path = os.path.join(self.dataset, "index.csv")
        self.df = pl.read_csv(csv_path, has_header=True, separator="	")
        if bands == "visu":
            self.df = self.df.select(["b2b3b4b8_10m", "b2b3b4b8_05m"])
            self.p0 = "b2b3b4b8_10m"
            self.p1 = "b2b3b4b8_05m"
        
        assert patch_size <= 256, "Patch size must be less than or equal to 256"
        if patch_size < 256 and patch_size > 0:
            #TODO: implement random cropping

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item_path = self.df[idx]
        p1 = item_path[self.p0].to_numpy()[0]
        p2 = item_path[self.p1].to_numpy()[0]

        p1 = os.path.join(self.dataset, p1)
        p2 = os.path.join(self.dataset, p2)

        # Load the images using rasterio
        with rasterio.open(p1) as src1:
            img1 = src1.read()  # Read all bands
        with rasterio.open(p2) as src2:
            img2 = src2.read()  # Read all bands

        # Normalize the images
        img1 = torch.tensor(img1, dtype=torch.float32) / 255.0
        img2 = torch.tensor(img2, dtype=torch.float32) / 255.0

        return img1, img2
