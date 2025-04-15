import torch
from torch.utils.data import Dataset
import rasterio
import os
import numpy as np
import polars as pl

class FloodDataset(Dataset):
    def __init__(self, patch_size=64):
        super(FloodDataset, self).__init__()
        self.patch_size = patch_size
        self.patches = []
        self.precompute_patches()

    def precompute_patches(self):
        root = os.listdir("/scratch/disc/e.bardet/Simple-VAE-RS/floods")
        for path in root:
            img_paths = [os.path.join("/scratch/disc/e.bardet/Simple-VAE-RS/floods", path, "S2", x) for x in os.listdir(os.path.join("/scratch/disc/e.bardet/Simple-VAE-RS/floods", path, "S2")) if x.endswith(".tif")]
            for img_path in img_paths:
                with rasterio.open(img_path) as src:
                    img = src.read()  # Read all bands
                height, width = img.shape[1], img.shape[2]
                for row in range(0, height, self.patch_size):
                    for col in range(0, width, self.patch_size):
                        if row + self.patch_size <= height and col + self.patch_size <= width:
                            patch = img[:, row:row + self.patch_size, col:col + self.patch_size]
                            quantiles = np.quantile(patch, [0.01, 0.99], axis=(1, 2), keepdims=True)
                            patch = (patch - quantiles[0]) / (quantiles[1] - quantiles[0]+1e-5)
                            patch = np.clip(patch, 0,1)
                            patch = torch.tensor(patch, dtype=torch.float32)
                            if not torch.isnan(patch).any():
                                self.patches.append(patch)

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        return self.patches[idx]

class Sen2VenDataset(Dataset):
    def __init__(self, csv_path="./ARM/index.csv", bands="visu"):
        super(Sen2VenDataset, self).__init__()
        self.df = pl.read_csv(csv_path, has_header=True, separator="	")
        print(self.df.get_columns())
        if bands == "visu":
            self.df = self.df.select(['b2b3b4b8_10m', 'b2b3b4b8_05m'])
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        pass

if __name__ == "__main__":
    dataset = Sen2VenDataset()
    print(dataset.df)
    print(len(dataset))