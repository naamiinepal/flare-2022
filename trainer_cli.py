from pytorch_lightning.utilities.cli import LightningCLI

from datamodules.datamodule import DataModule
from models.segmentor import Segmentor

cli = LightningCLI(
    Segmentor,
    DataModule,
    save_config_callback=None,  # Uncomment for wandb
    parser_kwargs={"parser_mode": "omegaconf"},
)
