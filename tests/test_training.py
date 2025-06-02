import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

import models.base as base_module
from models.cond_vae import Cond_SRVAE
from models.vae import VAE


class DummyWandbRun:
    def log(self, *args, **kwargs):
        pass

    def finish(self):
        pass


@pytest.fixture(autouse=True)
def dummy_wandb(monkeypatch):
    # force all wandb.init calls to return a no-op run
    monkeypatch.setattr(
        base_module.wandb, "init", lambda *args, **kwargs: DummyWandbRun()
    )


def test_vae_training_loop_runs_one_epoch():
    x_data = torch.randn(2, 4, 8, 8)
    ds = TensorDataset(x_data, x_data)
    loader = DataLoader(ds, batch_size=2)
    model = VAE(latent_size=128, patch_size=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # should finish without error
    model.fit(
        train_loader=loader,
        val_loader=loader,
        device="cpu",
        optimizer=optimizer,
        epochs=1,
        start_epoch=1,
        val_metrics_every=1,
        slurm_job_id="test",
    )
    assert model.scheduler.last_epoch == 1


def test_cond_vae_training_loop_runs_one_epoch():
    x_data = torch.randn(2, 4, 8, 8)
    y_data = torch.randn(2, 4, 4, 4)
    ds = TensorDataset(y_data, x_data)
    loader = DataLoader(ds, batch_size=2)
    model = Cond_SRVAE(latent_size=128, patch_size=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.fit(
        train_loader=loader,
        val_loader=loader,
        device="cpu",
        optimizer=optimizer,
        epochs=1,
        start_epoch=1,
        val_metrics_every=1,
        slurm_job_id="test",
    )
    assert model.scheduler.last_epoch == 1
