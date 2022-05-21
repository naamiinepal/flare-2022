from typing import Optional, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.transforms import AsDiscrete

from datamodule import FlareDataModule

NdarrayOrTensor = Union[np.ndarray, torch.Tensor]


class Segmentor(pl.LightningModule):
    def __init__(
        self,
        model_channels: Tuple[int, ...] = (16, 32, 64, 128, 256),
        model_strides: Tuple[int, ...] = (2, 2, 2, 2),
        learning_rate: float = 1e-4,
        sw_batch_size: int = 4,
        sw_overlap: float = 0.1,
        plateu_patience: int = 2,
        plateu_factor: float = 0.1,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=FlareDataModule.NUM_LABELS + 1,
            channels=model_channels,
            strides=model_strides,
            num_res_units=2,
            norm="batch",
            bias=False,  # no need for bias for batch norm
        )

        self.criterion = DiceLoss(to_onehot_y=True, softmax=True)

        self.post_pred = AsDiscrete(argmax=True, to_onehot=self.model.out_channels)
        self.post_label = AsDiscrete(to_onehot=self.model.out_channels)

        self.dice_metric = DiceMetric(include_background=False)

    def forward(self, image):
        return self.model(image)

    def training_step(self, batch, batch_idx):
        label = batch["label"]
        image = batch["image"]

        output = self(image)

        self.compute_dice_score(output, label)

        # self.log("train_dice_score", score, prog_bar=True)

        loss = self.criterion(output, label)

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        label = batch["label"]
        image = batch["image"]

        output = sliding_window_inference(
            image,
            self.roi_size,
            self.hparams.sw_batch_size,
            self,
            overlap=self.hparams.sw_overlap,
        )

        self.compute_dice_score(output, label)

        loss = self.criterion(output, label)

        self.log("val_loss", loss, batch_size=1)

    def validation_epoch_end(self, outputs):
        score = self.dice_metric.aggregate()
        self.dice_metric.reset()
        self.log("val_dice_score", score, prog_bar=True)

    def compute_dice_score(self, output: NdarrayOrTensor, label: NdarrayOrTensor):
        output = tuple(map(self.post_pred, output))
        labels = tuple(map(self.post_label, label))
        self.dice_metric(output, labels)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), self.hparams.learning_rate
        )

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=self.hparams.plateu_factor,
                patience=self.hparams.plateu_patience,
            ),
            "monitor": "val_loss",
        }

        return [optimizer], [scheduler]

    def setup(self, stage: Optional[str] = None):
        if stage is None or stage == "fit":
            self.roi_size = self.trainer.datamodule.hparams.roi_size
