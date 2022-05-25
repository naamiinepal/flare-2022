from typing import Optional, Union

import numpy as np
import pytorch_lightning as pl
import torch
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from torch import nn

from saver import NiftiSaver

NdarrayOrTensor = Union[np.ndarray, torch.Tensor]


class Segmentor(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-4,
        sw_batch_size: int = 4,
        sw_overlap: float = 0.1,
        plateu_patience: int = 2,
        plateu_factor: float = 0.1,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters(ignore="model")
        self.model = model

        self.criterion = DiceLoss(to_onehot_y=True, softmax=True)

        self.post_pred = AsDiscrete(argmax=True, to_onehot=model.out_channels)
        self.post_label = AsDiscrete(to_onehot=model.out_channels)

        self.dice_metric = DiceMetric(include_background=False)

    def forward(self, image):
        return self.model(image)

    def training_step(self, batch, batch_idx):
        label = batch["label"]
        image = batch["image"]

        output = self(image)

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

        self.log("val_loss", loss, batch_size=1, prog_bar=True)

    def validation_epoch_end(self, outputs):
        score = self.dice_metric.aggregate()
        self.dice_metric.reset()
        self.log("val_dice_score", score, prog_bar=True)

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        image = batch["image"]

        output = sliding_window_inference(
            image,
            self.roi_size,
            self.hparams.sw_batch_size,
            self,
            overlap=self.hparams.sw_overlap,
            device="cpu",
        )

        batch_meta_data = {
            k: v.cpu() if isinstance(v, torch.Tensor) else v
            for k, v in batch["image_meta_dict"].items()
        }

        for i, out in enumerate(output):
            argmax_out = out.argmax(dim=0)
            meta_data = {k: v[i] for k, v in batch_meta_data.items()}
            self.saver(argmax_out, meta_data)

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
        self.roi_size = self.trainer.datamodule.hparams.roi_size
        if stage == "predict":
            self.saver = NiftiSaver(
                "rabinadk1", output_postfix="", separate_folder=False, print_log=False
            )
