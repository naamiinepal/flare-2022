from pytorch_lightning.cli import LightningCLI

from datamodules import BaseDataModule
from models.coarse_model import CoarseModel


cli = LightningCLI(
    CoarseModel,
    BaseDataModule,
    save_config_callback=None,  # Uncomment for wandb
    parser_kwargs={"parser_mode": "omegaconf"},
)
