#!.venv/bin/python

from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.cli import LightningCLI

# Removing datamodule_class doesn't work for the data
cli = LightningCLI(
    datamodule_class=LightningDataModule,
    subclass_mode_data=True,
    save_config_callback=None,  # Uncomment for wandb
    parser_kwargs={"parser_mode": "omegaconf"},
)
