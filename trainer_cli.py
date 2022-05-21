from pytorch_lightning.utilities.cli import LightningCLI

from datamodule import FlareDataModule
from model import Segmentor

cli = LightningCLI(Segmentor, FlareDataModule, save_config_callback=None)
