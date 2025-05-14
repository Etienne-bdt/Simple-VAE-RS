import argparse
import os
import time

import torch
from tqdm import tqdm

from dataset import init_dataloader
from loss import cond_loss
from model import Cond_SRVAE
from test import test
from utils import EarlyStopper, SrEvaluator


def train(
    device,
    model,
    train_loader,
    val_loader,
    gamma,
    gamma2,
    optimizer,
    epochs,
    start_epoch=1,
    pretrain=False,
    bands=None,
    **kwargs,
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
    bands = bands or [2, 1, 0]  # Default to RGB bands if not provided
    # SLURM JOB ID
    slurm_job_id = os.environ.get(
        "SLURM_JOB_ID", f"local_{time.strftime('%Y%m%D-%H%M%S')}"
    )
    best_loss = float("inf")  # Initialize best loss to infinity
    early_stopper = EarlyStopper(patience=20, delta=0.001)  # Initialize early stopper
    print("Computing baseline...")
    evaluator = SrEvaluator(val_loader, start_epoch)  # Initialize evaluator
    writer = evaluator.writer  # Get the TensorBoard writer from the evaluator

    print("Baseline computed.")
    for _batch in val_loader:
        pass
    y_val, x_val = _batch
    _, c, h, w = y_val.shape

    evaluator.log_images(y_val[:4, bands, :, :], "Reconstruction/LR_Original", 0)
    evaluator.log_images(x_val[:4, bands, :, :], "Reconstruction/HR_Original", 0)

    for epoch in range(start_epoch, epochs + 1):
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
        val_recon_ssim_lr, val_recon_lpips_lr, val_recon_ssim_hr, val_recon_lpips_hr = (
            0,
            0,
            0,
            0,
        )
        val_tot_ssim, val_tot_lpips = 0, 0
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
                if epoch % kwargs["val_metrics_every"] == 0:
                    conditional_gen = model.conditional_generation(y)
                    ssim, lpips = evaluator.compute_metrics(conditional_gen, x)
                    val_tot_ssim, val_tot_lpips = (
                        val_tot_ssim + ssim.item(),
                        val_tot_lpips + lpips,
                    )
                    ssim_lr, lpips_lr = evaluator.compute_metrics(y_hat, y)
                    ssim_hr, lpips_hr = evaluator.compute_metrics(x_hat, x)
                    (
                        val_recon_ssim_lr,
                        val_recon_lpips_lr,
                        val_recon_ssim_hr,
                        val_recon_lpips_hr,
                    ) = (
                        val_recon_ssim_lr + ssim_lr.item(),
                        val_recon_lpips_lr + lpips_lr,
                        val_recon_ssim_hr + ssim_hr.item(),
                        val_recon_lpips_hr + lpips_hr,
                    )

        print(f"====> Validation loss: {(val_loss) / len(val_loader.dataset):.4f}")

        if early_stopper(val_loss / len(val_loader.dataset)):
            print(
                f"====> Early stopping at epoch {epoch} with loss: {val_loss / len(val_loader.dataset):.4f}"
            )
            break
        if val_loss / len(val_loader.dataset) < best_loss:
            best_loss = val_loss / len(val_loader.dataset)
            print(
                f"====> New best model found at epoch {epoch} with loss: {best_loss:.4f}"
            )
            # Save training data
            save_dict = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "gamma": gamma,
                "gamma2": gamma2,
            }
            torch.save(
                save_dict,
                f"{'pre_' if pretrain else ''}best_model_{slurm_job_id}.pth",
            )

        # Log reconstruction and Conditional generation

        evaluator.log_images(
            y_hat.view(-1, c, h, w)[:4, bands, :, :], "Reconstruction/LR", epoch
        )
        evaluator.log_images(
            x_hat.view(-1, c, h * 2, w * 2)[:4, bands, :, :], "Reconstruction/HR", epoch
        )
        if not pretrain:
            evaluator.log_images(
                conditional_gen.view(-1, c, h * 2, w * 2)[:4, bands, :, :],
                "Conditional Generation/HR",
                epoch,
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
        if epoch % kwargs["val_metrics_every"] == 0:
            writer.add_scalars(
                "Metrics/SSIM",
                {
                    "Recon_LR": val_recon_ssim_lr / len(val_loader.dataset),
                    "Recon_HR": val_recon_ssim_hr / len(val_loader.dataset),
                    "SR": val_tot_ssim / len(val_loader.dataset),
                },
                epoch,
            )
            writer.add_scalars(
                "Metrics/LPIPS",
                {
                    "Recon_LR": val_recon_lpips_lr / len(val_loader.dataset),
                    "Recon_HR": val_recon_lpips_hr / len(val_loader.dataset),
                    "SR": val_tot_lpips / len(val_loader.dataset),
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
    latent_size = 4096
    model = Cond_SRVAE(latent_size, args.patch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if args.model_ckpt:
        print("Loading model from checkpoint...")
        save_dict = torch.load(args.model_ckpt)
        start_epoch = save_dict["epoch"] + 1
        model.load_state_dict(save_dict["model_state_dict"])
        print("Model loaded successfully.")
        print("Loading optimizer state...")
        gamma = save_dict["gamma"]
        gamma2 = save_dict["gamma2"]
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
        optimizer.add_param_group({"params": [gamma, gamma2]})
        optimizer.load_state_dict(save_dict["optimizer_state_dict"])
        print("Optimizer state loaded successfully.")
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
        start_epoch = 1
        gamma = torch.tensor([1.0]).to(device)
        gamma2 = torch.tensor([1.0]).to(device)
        gamma.requires_grad = True
        gamma2.requires_grad = True
        optimizer.add_param_group({"params": [gamma, gamma2]})
    # Lower learning rate for the conditional part
    for param_group in optimizer.param_groups:
        param_group["lr"] = 5e-4

    if not (args.test and args.model_ckpt):
        train(
            device,
            model,
            train_loader,
            val_loader,
            gamma,
            gamma2,
            optimizer,
            epochs=args.epochs,
            start_epoch=start_epoch,
            pretrain=False,
        )

    test(device, model, val_loader)


def parse_args():
    """
    Parse command line arguments. Notably to set the number of epochs or change the dataset.
    """
    parser = argparse.ArgumentParser(description="Train a VAE model.")
    parser.add_argument(
        "--pre_epochs",
        type=int,
        default=20,
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

    parser.add_argument(
        "--model_ckpt",
        type=str,
        help="Path to the model checkpoint to resume training.",
    )

    parser.add_argument(
        "--val_metrics_every",
        type=int,
        default=5,
        help="Number of epochs between validation metrics computation.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    print("==========================")
    print("Initializing training with the following arguments:")
    print(arguments)
    print("--------------------------")
    print(
        f"Model checkpoint: {'not' if arguments.model_ckpt is None else arguments.model_ckpt} provided"
    )
    if arguments.model_ckpt:
        print("Checking if model exists...")
        if not os.path.exists(arguments.model_ckpt):
            raise FileNotFoundError(
                f"Model checkpoint {arguments.model_ckpt} not found."
            )
        else:
            print("Model checkpoint found.")
    print("--------------------------")
    print("Device:", "cuda" if torch.cuda.is_available() else "cpu")
    print("==========================")
    main(args=arguments)
