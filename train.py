import argparse

import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from dataset import init_dataloader
from loss import cond_loss
from model import Cond_SRVAE
from utils import normalize_image


def train(
    device,
    model,
    train_loader,
    val_loader,
    gamma,
    gamma2,
    optimizer,
    epochs,
    pretrain=False,
    bands=[2, 1, 0],
):
    """
    Training script for the Conditional SRVAE model.
    Args:
        device: The device to use for training (CPU or GPU).
        model: The Conditional SRVAE model.
        train_loader: DataLoader for the training set.
        val_loader: DataLoader for the validation set.
        gamma: The gamma parameter for the loss function.
        gamma2: The gamma2 parameter for the loss function.
        optimizer: The optimizer for training.
        epochs: Number of epochs to (pre)train the model.
        pretrain: If True, pretrain the model on low resolution data.
        bands: List of bands to use for visualization. (Default is the usual Visual RGB bands)
    """
    writer = SummaryWriter()  # Initialize TensorBoard writer
    best_loss = float("inf")  # Initialize best loss to infinity
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        tot_mse_x, tot_kld_u, tot_mse_y, tot_kld_z = (0, 0, 0, 0)
        for _, batch in tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Training, Epoch {epoch}/{epochs}",
            unit="batch",
        ):
            optimizer.zero_grad()
            y, x = batch
            y, x = y.to(device), x.to(device)
            x_hat, y_hat, mu_z, logvar_z, mu_u, logvar_u, mu_z_uy, logvar_z_uy = model(
                x, y
            )
            mse_x, kld_u, mse_y, kld_z = cond_loss(
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
                gamma,
                gamma2,
            )
            loss = (
                mse_x + kld_u + mse_y + kld_z if not pretrain else mse_y + kld_u + mse_x
            )
            tot_kld_u, tot_kld_z, tot_mse_x, tot_mse_y = (
                tot_kld_u + kld_u.item(),
                tot_kld_z + kld_z.item(),
                tot_mse_x + mse_x.item(),
                tot_mse_y + mse_y.item(),
            )
            loss.backward()
            train_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Append train losses to lists after each epoch

        print(
            f"====> Epoch: {epoch} Average loss: {(train_loss) / len(train_loader.dataset):.4f}"
        )

        val_loss = 0
        val_tot_kld_u, val_tot_kld_z, val_tot_mse_x, val_tot_mse_y = (0, 0, 0, 0)
        model.eval()
        with torch.no_grad():
            for _, batch in enumerate(val_loader):
                y, x = batch
                y, x = y.to(device), x.to(device)
                x_hat, y_hat, mu_z, logvar_z, mu_u, logvar_u, mu_z_uy, logvar_z_uy = (
                    model(x, y)
                )
                v_mse_x, v_kld_u, v_mse_y, v_kld_z = cond_loss(
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
                    gamma,
                    gamma2,
                )
                v_loss = (
                    v_mse_x + v_kld_u + v_mse_y + v_kld_z
                    if not pretrain
                    else v_mse_y + v_kld_u + v_mse_x
                )
                val_tot_kld_u, val_tot_kld_z, val_tot_mse_x, val_tot_mse_y = (
                    val_tot_kld_u + v_kld_u.item(),
                    val_tot_kld_z + v_kld_z.item(),
                    val_tot_mse_x + v_mse_x.item(),
                    val_tot_mse_y + v_mse_y.item(),
                )
                val_loss += v_loss.item()

        print(f"====> Validation loss: {(val_loss) / len(val_loader.dataset):.4f}")

        if val_loss / len(val_loader.dataset) < best_loss:
            best_loss = val_loss / len(val_loader.dataset)
            print(
                f"====> New best model found at epoch {epoch} with loss: {best_loss:.4f}"
            )
            torch.save(model.state_dict(), "best_model.pth")

        # Log reconstruction and Conditional generation

        writer.add_images(
            "Reconstruction/LR_Original",
            normalize_image(y.view(-1, 4, 128, 128)[:, bands, :, :]),
            global_step=epoch,
            dataformats="NCHW",
        )

        writer.add_images(
            "Reconstruction/LR",
            normalize_image(y_hat.view(-1, 4, 128, 128)[:, bands, :, :]),
            global_step=epoch,
            dataformats="NCHW",
        )

        conditional_gen = model.conditional_generation(y)
        writer.add_images(
            "Conditional Generation/Original",
            normalize_image(y.view(-1, 4, 128, 128)[:, bands, :, :]),
            global_step=epoch,
            dataformats="NCHW",
        )

        writer.add_images(
            "Conditional Generation/HR",
            normalize_image(conditional_gen.view(-1, 4, 256, 256)[:, bands, :, :]),
            global_step=epoch,
            dataformats="NCHW",
        )

        writer.add_images(
            "Reconstruction/HR",
            normalize_image(x_hat.view(-1, 4, 256, 256)[:, bands, :, :]),
            global_step=epoch,
            dataformats="NCHW",
        )

        # Log to TensorBoard
        writer.add_scalars(
            "Loss/KLD_u",
            {
                "Train": tot_kld_u / len(train_loader.dataset),
                "Validation": val_tot_kld_u / len(val_loader.dataset),
            },
            epoch,
        )
        writer.add_scalars(
            "Loss/KLD_z",
            {
                "Train": tot_kld_z / len(train_loader.dataset),
                "Validation": val_tot_kld_z / len(val_loader.dataset),
            },
            epoch,
        )
        writer.add_scalars(
            "Loss/MSE_y",
            {
                "Train": tot_mse_y / len(train_loader.dataset),
                "Validation": val_tot_mse_y / len(val_loader.dataset),
            },
            epoch,
        )
        writer.add_scalars(
            "Loss/MSE_x",
            {
                "Train": tot_mse_x / len(train_loader.dataset),
                "Validation": val_tot_mse_x / len(val_loader.dataset),
            },
            epoch,
        )
        writer.add_scalars(
            "Loss/Total",
            {
                "Train": train_loss / len(train_loader.dataset),
                "Validation": val_loss / len(val_loader.dataset),
            },
            epoch,
        )
        writer.add_scalars(
            "Gamma", {"Gamma_y": gamma2.item(), "Gamma_x": gamma.item()}, epoch
        )

    if torch.isnan(loss):
        raise ValueError("Loss is NaN, stopping training.")

    writer.close()  # Close the TensorBoard writer
    return


def main(args):
    """
    args : arguments from the command line
    args.epochs : number of epochs to train the model
    args.dataset : dataset to use for training
    """
    train_loader, val_loader = init_dataloader(
        args.dataset, args.batch_size, args.patch_size
    )
    latent_size = 3500
    model = Cond_SRVAE(latent_size, args.patch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    gamma = torch.tensor([0.25]).to(device)
    gamma.requires_grad = True
    gamma2 = torch.tensor([0.25]).to(device)
    gamma2.requires_grad = True
    optimizer.add_param_group({"params": [gamma, gamma2]})

    model.freeze_cond()

    train(
        device,
        model,
        train_loader,
        val_loader,
        gamma,
        gamma2,
        optimizer,
        epochs=args.pre_epochs,
        pretrain=True,
    )

    model.unfreeze_cond()

    train(
        device,
        model,
        train_loader,
        val_loader,
        gamma,
        gamma2,
        optimizer,
        epochs=args.epochs,
        pretrain=False,
    )


def parse_args():
    """
    Parse command line arguments. Notably to set the number of epochs or change the dataset.
    """
    parser = argparse.ArgumentParser(description="Train a VAE model.")
    parser.add_argument(
        "--pre_epochs",
        type=int,
        default=5,
        help="Number of epochs to pre-train the low resolution model.",
    )
    parser.add_argument(
        "--epochs", type=int, default=200, help="Number of epochs to train the model."
    )
    parser.add_argument(
        "--dataset", type=str, default="s2v", help="Type of the dataset"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training and validation.",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=64,
        help="Patch size of the High-Res Images.",
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="If set, the model will be tested instead of trained.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    print("==========================")
    print("Initializing training with the following arguments:")
    print(arguments)
    print("==========================")
    main(args=arguments)
