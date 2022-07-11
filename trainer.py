#!.venv/bin/python

from pytorch_lightning.utilities.cli import LightningCLI

from models import BaseModel
from datamodules import BaseDataModule

# Removing datamodule_class doesn't work for the data
cli = LightningCLI(
    model_class=BaseModel,
    datamodule_class=BaseDataModule,
    subclass_mode_model=True,
    subclass_mode_data=True,
    save_config_callback=None,  # Uncomment for wandb
    parser_kwargs={"parser_mode": "omegaconf"},
)
