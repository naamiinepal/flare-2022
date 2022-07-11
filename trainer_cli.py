from pytorch_lightning.utilities.cli import LightningCLI

from datamodules.single_step_datamodule import SingleStepDataModule
from models.single_step_model import SingleStepModel

cli = LightningCLI(
    SingleStepModel,
    SingleStepDataModule,
    save_config_callback=None,  # Uncomment for wandb
    parser_kwargs={"parser_mode": "omegaconf"},
)
