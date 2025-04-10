import torch 
from model import VAE
from loss import loss_function
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from dataset import FloodDataset
from torch.utils.data import DataLoader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        for batch_idx, data in tqdm(enumerate(train_loader)):
            data = data.to(device)
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
        for batch_idx, data in enumerate(val_loader):
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

def init_dataloader():
    ds = FloodDataset(patch_size=256)
    train_size = int(0.8 * len(ds))
    val_size = len(ds) - train_size
    train_ds, val_ds = torch.utils.data.random_split(ds, [train_size, val_size])
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=6)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=6)
    return train_loader, val_loader

def main():
    train_loader , val_loader = init_dataloader()
    print(f"Train dataset size: {len(train_loader.dataset)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_size = 4096
    model = VAE(latent_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    gamma = torch.tensor([0.75]).to(device)
    gamma.requires_grad = True
    optimizer.add_param_group({'params': gamma})
    print("Training the model...")
    t_mse, t_kld, v_mse, v_kld, gamma_vals = train(model, train_loader, val_loader, gamma, optimizer, epochs=200)

    # Save the model
    torch.save(model.state_dict(), 'vae_model.pth')

    

if __name__ == "__main__":
    print("Starting training...")
    main()