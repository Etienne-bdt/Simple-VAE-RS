import argparse

import torch
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from dataset import init_dataloader
from loss import cond_loss
from model import Cond_SRVAE


def train(device, model, train_loader, val_loader, gamma, gamma2, optimizer, epochs):
    writer = SummaryWriter()  # Initialize TensorBoard writer
    best_loss = float("inf")  # Initialize best loss to infinity
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for _, batch in tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc="Training",
            unit="batch",
        ):
            y, x = batch
            y, x = y.to(device), x.to(device)
            optimizer.zero_grad()
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
            loss = mse_x + kld_u + mse_y + kld_z
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        # Append train losses to lists after each epoch

        print(
            f"====> Epoch: {epoch} Average loss: {(train_loss) / len(train_loader.dataset):.4f}"
        )

        val_loss = 0
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
                v_loss = v_mse_x + v_kld_u + v_mse_y + v_kld_z
                val_loss += v_loss.item()

        print(f"====> Validation loss: {(val_loss) / len(val_loader.dataset):.4f}")

        if val_loss / len(val_loader.dataset) < best_loss:
            best_loss = val_loss / len(val_loader.dataset)
            print(
                f"====> New best model found at epoch {epoch} with loss: {best_loss:.4f}"
            )
            torch.save(model.state_dict(), "best_model.pth")

        # Log validation losses to TensorBoard
        writer.add_scalars(
            "Loss/KLD_u",
            {
                "Train": kld_u.item() / len(val_loader.dataset),
                "Validation": v_kld_u.item() / len(val_loader.dataset),
            },
            epoch,
        )
        writer.add_scalars(
            "Loss/KLD_z",
            {
                "Train": kld_z.item() / len(val_loader.dataset),
                "Validation": v_kld_z.item() / len(val_loader.dataset),
            },
            epoch,
        )
        writer.add_scalars(
            "Loss/MSE_y",
            {
                "Train": mse_y.item() / len(val_loader.dataset),
                "Validation": v_mse_y.item() / len(val_loader.dataset),
            },
            epoch,
        )
        writer.add_scalars(
            "Loss/MSE_x",
            {
                "Train": mse_x.item() / len(val_loader.dataset),
                "Validation": v_mse_x.item() / len(val_loader.dataset),
            },
            epoch,
        )
        writer.add_scalars(
            "Loss/Total",
            {
                "Train": loss / len(val_loader.dataset),
                "Validation": v_loss / len(val_loader.dataset),
            },
            epoch,
        )
        writer.add_scalars(
            "Gamma", {"Gamma_y": gamma2.item(), "Gamma_x": gamma.item()}, epoch
        )
    writer.close()  # Close the TensorBoard writer
    return


def main(args):
    """
    args : arguments from the command line
    args.epochs : number of epochs to train the model
    args.dataset : dataset to use for training
    """
    train_loader, val_loader = init_dataloader(args.dataset)
    latent_size = 2000
    model = Cond_SRVAE(latent_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    gamma = torch.tensor(0.5, requires_grad=True).to(device)
    gamma2 = torch.tensor(0.5, requires_grad=True).to(device)

    train(
        device,
        model,
        train_loader,
        val_loader,
        gamma,
        gamma2,
        optimizer,
        epochs=args.epochs,
    )


def parse_args():
    """
    Parse command line arguments. Notably to set the number of epochs or change the dataset.
    """
    parser = argparse.ArgumentParser(description="Train a VAE model.")
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of epochs to train the model."
    )
    parser.add_argument(
        "--dataset", type=str, default="s2v", help="Type of the dataset"
    )

    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    main(args=arguments)
