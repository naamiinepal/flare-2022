from pytorch_lightning.utilities.cli import LightningCLI

from datamodule import DataModule
from segmentor import Segmentor

cli = LightningCLI(
    Segmentor,
    DataModule,
    save_config_callback=None,
    parser_kwargs={"parser_mode": "omegaconf"},
)
