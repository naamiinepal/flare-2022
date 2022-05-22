from pytorch_lightning.utilities.cli import LightningCLI

from datamodule import DataModule
from model import Segmentor

cli = LightningCLI(
    Segmentor,
    DataModule,
    save_config_callback=None,
    parser_kwargs={"parser_mode": "omegaconf"},
)
