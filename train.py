import argparse
import os
import time

import torch

import callbacks
import models
from dataset import init_dataloader
from task import sr_task


def main(args):
    """
    args : arguments from the command line
    args.epochs : number of epochs to train the model
    args.dataset : dataset to use for training
    """
    train_loader, val_loader = init_dataloader(
        args.dataset, args.batch_size, args.patch_size
    )
    latent_size = args.latent_size
    if latent_size <= 0:
        raise ValueError("Latent size must be a positive integer.")
    slurm_job_id = os.environ.get(
        "SLURM_JOB_ID", f"local_{time.strftime('%Y%m%D-%H%M%S')}"
    )
    results_dir = os.path.join("results", str(latent_size), slurm_job_id)
    os.makedirs(results_dir, exist_ok=True)

    callbacks_list = [
        callbacks.ModelCheckpoint(slurm_job_id, "ckpt", monitor="val_loss", mode="min"),
        callbacks.EarlyStopping(patience=25, delta=0.01),
    ]
    if args.model_type == "VAE":
        model = models.VAE(latent_size, args.patch_size // 2, callbacks=callbacks_list)
    elif args.model_type == "Cond_SRVAE":
        model = models.Cond_SRVAE(
            latent_size, args.patch_size, callbacks=callbacks_list
        )

    else:
        raise ValueError(
            f"Unknown model type: {args.model_type}. Choose 'Cond_SRVAE' or 'VAE'."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if args.model_ckpt:
        print("Loading model from checkpoint...")
        save_dict = torch.load(args.model_ckpt)
        start_epoch = save_dict["epoch"] + 1
        model.load_state_dict(save_dict["model_state_dict"])
        print("Model loaded successfully.")
        print("Loading optimizer state...")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizer.load_state_dict(save_dict["optimizer_state_dict"])
        print("Optimizer state loaded successfully.")
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        start_epoch = 1

    # Lower learning rate for the conditional part
    for param_group in optimizer.param_groups:
        param_group["lr"] = 1e-3

    if not (args.test and args.model_ckpt):
        model.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            device=device,
            optimizer=optimizer,
            start_epoch=start_epoch,
            val_metrics_every=args.val_metrics_every,
            slurm_job_id=slurm_job_id,
        )

    sr_task(device, model, val_loader)


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

    parser.add_argument(
        "-l", "--latent_size", type=int, default=10000, help="Size of the latent space."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="Cond_SRVAE",
        choices=["Cond_SRVAE", "VAE"],
        help="Model to use : 'Cond_SRVAE' ou 'VAE'",
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
