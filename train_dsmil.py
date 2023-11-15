import os
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
import argparse
import warnings
from datasets.data_interface import DataInterface
from dsmil import DSMIL

warnings.filterwarnings("ignore")


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def get_args():
    parser = argparse.ArgumentParser(description="Configuration for training a model.")

    # General settings
    parser.add_argument(
        "--comment", type=str, default="", help="General comment about the run."
    )
    parser.add_argument("--seed", type=int, default=2021, help="Random seed.")
    parser.add_argument(
        "--fp16", action="store_true", help="Enable mixed precision training."
    )
    parser.add_argument(
        "--amp_level",
        type=str,
        default="O2",
        choices=["O1", "O2"],
        help="AMP optimization level.",
    )
    parser.add_argument("--precision", type=int, default=16, help="Precision level.")
    parser.add_argument(
        "--multi_gpu_mode",
        type=str,
        default="dp",
        choices=["dp", "ddp"],
        help="Multi-GPU mode.",
    )
    parser.add_argument(
        "--gpus", nargs="+", type=int, default=[0], help="GPU IDs to use."
    )
    parser.add_argument(
        "--epochs", type=int, default=200, help="Number of training epochs."
    )
    parser.add_argument(
        "--grad_acc", type=int, default=2, help="Gradient accumulation steps."
    )
    parser.add_argument(
        "--frozen_bn", action="store_true", help="Freeze batch normalization layers."
    )
    parser.add_argument(
        "--patience", type=int, default=10, help="Patience for early stopping."
    )
    parser.add_argument(
        "--server",
        type=str,
        default="test",
        choices=["train", "test"],
        help="Server mode.",
    )
    parser.add_argument(
        "--log_path", type=str, default="logs_v2/", help="Path for logging."
    )

    # Data settings
    parser.add_argument(
        "--dataset_name", type=str, default="camel_data", help="Name of the dataset."
    )
    parser.add_argument("--data_shuffle", action="store_true", help="Shuffle the data.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/mnt/nvme0n1/ICCV/ds-mil/datasets/cam-16/",
        help="Directory of the data.",
    )
    parser.add_argument(
        "--label_dir",
        type=str,
        default="/mnt/nvme0n1/ICCV/camelyon_5_fold/",
        help="Directory of the labels.",
    )
    parser.add_argument("--fold", type=int, default=0, help="Current fold.")
    parser.add_argument("--nfold", type=int, default=4, help="Number of folds.")

    # Train dataloader settings
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size for training."
    )
    parser.add_argument(
        "--train_num_workers",
        type=int,
        default=8,
        help="Number of workers for training dataloader.",
    )

    # Test dataloader settings
    parser.add_argument(
        "--test_batch_size", type=int, default=1, help="Batch size for testing."
    )
    parser.add_argument(
        "--test_num_workers",
        type=int,
        default=8,
        help="Number of workers for testing dataloader.",
    )

    # Model settings
    parser.add_argument(
        "--model_name", type=str, default="TransMIL", help="Model name."
    )
    parser.add_argument("--n_classes", type=int, default=2, help="Number of classes.")

    # Optimizer settings
    parser.add_argument("--opt", type=str, default="lookahead_radam", help="Optimizer.")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate.")
    parser.add_argument("--opt_eps", default=None, help="Epsilon value for optimizer.")
    parser.add_argument(
        "--opt_betas", default=None, nargs=2, type=float, help="Betas for optimizer."
    )
    parser.add_argument(
        "--momentum", default=None, type=float, help="Momentum for optimizer."
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.00001, help="Weight decay."
    )

    # Loss settings
    parser.add_argument(
        "--base_loss", type=str, default="CrossEntropyLoss", help="Base loss function."
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="logs",
        help="Directory to save the model weights to",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use for training",
    )

    parser.add_argument(
        "--log_dir", type=str, default="./logs/", help="directory to save logs"
    )

    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default="logs/DMSO/epoch=0-step=0.ckpt",
        help="Path to pretrained model",
    )
    parser.add_argument(
        "--project_name",
        type=str,
        default="DSMIL_camel",
        help="Name of the project",
    )
    parser.add_argument(
        "--logger",
        type=str,
        default="wandb",
        choices=["wandb", "tensorboard", "neptune"],
        help="Whether to use wandb for logging",
    )
    parser.add_argument("--max_epochs", type=int, default=200, help="Number of epochs")

    return parser.parse_args()


def train(args):
    pl.seed_everything(42)

    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=0.00, patience=10, verbose=True, mode="min"
    )

    checkpoint_callback = ModelCheckpoint(monitor="val_loss")
    lr_monitor = LearningRateMonitor(logging_interval="step")

    DataInterface_dict = {
        "train_batch_size": args.train_batch_size,
        "train_num_workers": args.train_num_workers,
        "test_batch_size": args.test_batch_size,
        "test_num_workers": args.test_num_workers,
        "dataset_name": args.dataset_name,
        "dataset_cfg": args,
    }
    dm = DataInterface(**DataInterface_dict)
    dm.setup()

    setattr(
        args,
        "log_dir",
        os.path.join(args.log_dir, args.project_name, f"fold_{args.fold}"),
    )

    model = DSMIL(
        num_classes=2,
        criterion=nn.BCEWithLogitsLoss(),
        i_class="i_class",
        log_dir=args.log_dir,
    )

    if args.logger == "wandb":
        logger = WandbLogger(
            project=args.project_name,
            name=f"_fold_{args.fold}",
            log_model=True,
            save_dir=args.log_dir,
        )

    elif args.logger == "tensorboard":
        logger = pl.loggers.TensorBoardLogger(
            save_dir=args.log_dir,
            name=f"_fold_{args.fold}",
        )

    else:
        raise ValueError(f"Invalid logger {args.logger}")

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.num_gpus,
        max_epochs=args.max_epochs,
        callbacks=[early_stop_callback, checkpoint_callback, lr_monitor],
        default_root_dir=args.log_dir,
        logger=logger,
        num_sanity_val_steps=0,
    )
    trainer.fit(model, dm)
    trainer.test(model=model, datamodule=dm)


if __name__ == "__main__":
    print("Starting training")
    import warnings

    warnings.filterwarnings("ignore")

    args = get_args()
    train(args)
