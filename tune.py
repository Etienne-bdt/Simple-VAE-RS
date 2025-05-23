import os
import time

import optuna
import torch

from dataset import init_dataloader
from model import Cond_SRVAE
from train import train


def objective(trial):
    latent_size = trial.suggest_int("latent_size", 1024, 15000)
    train_loader, val_loader = init_dataloader("s2v", 2, 64)

    slurm_job_id = os.environ.get(
        "SLURM_JOB_ID", f"local_{time.strftime('%Y%m%D-%H%M%S')}"
    )
    model = Cond_SRVAE(latent_size, 64, num_comp=16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    start_epoch = 1
    gamma = torch.tensor([0.03]).to(device)
    gamma2 = torch.tensor([0.04]).to(device)
    gamma.requires_grad = True
    gamma2.requires_grad = True
    optimizer.add_param_group({"params": [gamma, gamma2]})
    return train(
        device,
        model,
        train_loader,
        val_loader,
        gamma,
        gamma2,
        optimizer,
        epochs=50,
        start_epoch=start_epoch,
        pretrain=False,
        slurm_job_id=slurm_job_id,
        no_save=True,
    )


if __name__ == "__main__":
    study = optuna.create_study(
        storage="sqlite:///db.sqlite3", study_name="srLatent", direction="maximize"
    )
    study.optimize(objective, n_trials=50)
