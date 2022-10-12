from pytorch_lightning.utilities.cli import LightningCLI

from datamodules.c2f_datamodule import C2FDataModule
from models.c2f import C2FSegmentor

cli = LightningCLI(
    C2FSegmentor,
    C2FDataModule,
    save_config_callback=None,  # Uncomment for wandb
    parser_kwargs={"parser_mode": "omegaconf"},
)
