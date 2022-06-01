from typing import Iterable, Optional

import pytorch_lightning as pl
import torch
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from torch import nn

from datamodule import DataModule


class Segmentor(pl.LightningModule):

    val_dice_metric = DiceMetric(include_background=False)
    criterion = DiceLoss(include_background=False, to_onehot_y=True, softmax=True)

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-4,
        sw_batch_size: int = 4,
        sw_overlap: float = 0.1,
        plateu_patience: int = 2,
        plateu_factor: float = 0.1,
        monitor: str = "val/loss",
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters(ignore="model")
        self.model = model

    def forward(self, image):
        return self.model(image)

    # Using custom or multiple metrics (default_hp_metric=False)
    def on_train_start(self):
        if self.logger:
            self.logger.log_hyperparams(
                self.hparams, {"val/loss": 0, "val/dice_score": 0}
            )

    def training_step(self, batch, batch_idx):
        label = batch["label"]
        image = batch["image"]

        output = self(image)

        loss = self.criterion(output, label)

        self.log("train/loss", loss, on_epoch=True, batch_size=self.batch_size)

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

        self.log("val/loss", loss, batch_size=1, prog_bar=True)

    def validation_epoch_end(self, outputs):
        score = self.val_dice_metric.aggregate()
        self.val_dice_metric.reset()
        self.log("val/dice_score", score, prog_bar=True)

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

    def compute_dice_score(self, output: Iterable, label: Iterable):
        output = tuple(map(self.post_pred, output))
        labels = tuple(map(self.post_label, label))
        self.val_dice_metric(output, labels)

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
            "monitor": self.hparams.monitor,
        }

        return [optimizer], [scheduler]

    def setup(self, stage: Optional[str] = None):
        datamodule: DataModule = self.trainer.datamodule
        self.roi_size = datamodule.hparams.roi_size
        if stage is None or stage == "predict":
            self.saver = datamodule.saver
        if stage is None or stage == "fit":
            self.example_input_array = torch.empty(1, 1, *self.roi_size)
            self.batch_size = datamodule.hparams.batch_size
        if stage is None or stage == "fit" or stage == "validate":
            num_labels_with_bg = datamodule.hparams.num_labels_with_bg
            self.post_pred = AsDiscrete(argmax=True, to_onehot=num_labels_with_bg)
            self.post_label = AsDiscrete(to_onehot=num_labels_with_bg)

    def save_scripted(self, path: str):
        torch.jit.script(self.model).save(path)
