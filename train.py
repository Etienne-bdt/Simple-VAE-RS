import argparse

import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from dataset import FloodDataset, Sen2VenDataset
from loss import loss_function
from model import Cond_SRVAE_Lightning, VAE_Lightning

import lightning as L
import lightning.pytorch.callbacks as clb

def train(model, train_loader, val_loader, gamma, optimizer, epochs):
    writer = SummaryWriter()  # Initialize TensorBoard writer
    t_mse = []
    t_kld = []
    v_mse = []
    v_kld = []
    gamma_vals = []

    for epoch in range(epochs):
        gamma_vals.append(gamma.item())
        model.train()
        train_loss_mse = 0
        train_loss_kld = 0
        train_loss = 0
        for _, data in tqdm(enumerate(train_loader)):
            _,y = data
            data = y.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            mse, kld = loss_function(recon_batch, data, mu, logvar, gamma)
            loss = mse + kld
            train_loss_kld += kld.item()
            train_loss_mse += mse.item()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        # Append train losses to lists after each epoch
        t_mse.append(train_loss_mse / len(train_loader.dataset))
        t_kld.append(train_loss_kld / len(train_loader.dataset))

        print(f"====> Epoch: {epoch} Average loss: {(train_loss) / len(train_loader.dataset):.4f}")

        # Log training losses to TensorBoard
        writer.add_scalar('Loss/Train_MSE', t_mse[-1], epoch)
        writer.add_scalar('Loss/Train_KLD', t_kld[-1], epoch)
        writer.add_scalar('Gamma', gamma.item(), epoch)

        val_loss_mse = 0
        val_loss_kld = 0
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for _, data in enumerate(val_loader):
                _,y  = data
                data = y.to(device)
                recon_batch, mu, logvar = model(data)
                mse, kld = loss_function(recon_batch, data, mu, logvar, gamma)
                v_loss = mse + kld
                val_loss_kld += kld.item()
                val_loss_mse += mse.item()
                val_loss += v_loss.item()
        # Append val losses to lists after each epoch
        v_mse.append(val_loss_mse / len(val_loader.dataset))
        v_kld.append(val_loss_kld / len(val_loader.dataset))

        print(f"====> Validation loss: {(val_loss) / len(val_loader.dataset):.4f}")

        v_mse.append(val_loss_mse / len(val_loader.dataset))
        v_kld.append(val_loss_kld / len(val_loader.dataset))

        print(f"====> Validation loss: {(val_loss) / len(val_loader.dataset):.4f}")

        # Log validation losses to TensorBoard
        writer.add_scalar('Loss/Val_MSE', v_mse[-1], epoch)
        writer.add_scalar('Loss/Val_KLD', v_kld[-1], epoch)
        writer.add_scalar('Loss/gamma', gamma_vals[-1], epoch)
    writer.close()  # Close the TensorBoard writer
    return t_mse, t_kld, v_mse, v_kld, gamma_vals

def init_dataloader(dataset:str):
    if dataset == "Sen2Venus" or dataset == "sen2venus" or dataset == "s2v":
        ds = Sen2VenDataset()

    elif dataset == "Floods" or dataset == "floods":
        ds = FloodDataset(patch_size=256)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    train_size = int(0.8 * len(ds))
    val_size = len(ds) - train_size
    train_ds, val_ds = torch.utils.data.random_split(ds, [train_size, val_size])
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=6)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=6)
    return train_loader, val_loader

def main(args):
    train_loader, val_loader = init_dataloader(args.dataset)
    latent_size = 1000
    model = Cond_SRVAE_Lightning(latent_size)

    """print("Training the model...")
    if os.path.exists('vae_model.pth'):
        model.load_state_dict(torch.load('vae_model.pth'))
        model.eval()
        print("Model loaded from file.")           
    else:
        train(model, train_loader, val_loader, gamma, optimizer, epochs=args.epochs)
        # Save the model
        torch.save(model.state_dict(), 'vae_model.pth')
    """

    trainer = L.Trainer(
        devices=1,
        num_nodes=1,
        accelerator="cuda",
        max_epochs=args.epochs,
        gradient_clip_val=5,
        callbacks=[
            clb.EarlyStopping(monitor="val_loss", patience=5, verbose=True),
        ],
    )

    trainer.fit(model, train_loader, val_loader)
    # Save the model

    """
    z_sample = torch.randn(1, latent_size).to(device)
    recon_sample = model.decode(z_sample)[0,[3,2,1],:,:].cpu().detach().permute(1,2,0).numpy()
    plt.imsave('sample_reconstruction.png', recon_sample, cmap='gray')"""

    """    _,data = next(iter(val_loader))
    recon_batch, mu, logvar = model(data)
    recon_batch = recon_batch[0,[3,2,1],:,:].cpu().detach().permute(1,2,0).numpy()
    plt.imsave('sample_reconstruction.png', recon_batch, cmap='gray')
    """


def parse_args():
    """
    Parse command line arguments. Notably to set the number of epochs or change the dataset.
    """
    parser = argparse.ArgumentParser(description="Train a VAE model.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train the model.")
    parser.add_argument("--dataset", type=str, default="s2v", help="Type of the dataset")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
