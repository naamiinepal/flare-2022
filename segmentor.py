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

    val_dice_metric = DiceMetric(include_background=False, reduction="mean_batch")

    common_criterion_kwargs = {
        "include_background": False,
        "to_onehot_y": True,
    }
    soft_criterion = DiceLoss(include_background=False, to_onehot_y=True, softmax=True)
    logit_criterion = DiceLoss(include_background=False, to_onehot_y=True)

    labels = (
        "liver",
        "right_kidney",
        "spleen",
        "pancreas",
        "aorta",
        "ivc",
        "rag",
        "lag",
        "gallbladder",
        "esophagus",
        "stomach",
        "duodenum",
        "left_kidney",
    )

    def __init__(
        self,
        model: nn.Module,
        checkpoint_path: Optional[str] = None,
        pseudo_threshold: float = 0.9,
        unsup_weight: float = 1,
        learning_rate: float = 0.03,
        sw_batch_size: int = 4,
        sw_overlap: float = 0.1,
        plateu_patience: int = 2,
        plateu_factor: float = 0.1,
        monitor: str = "val/loss",
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters(ignore="model")

        if checkpoint_path:
            model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
            print("\nLoaded Model Checkpoint\n")

        self.model = model

    def forward(self, image) -> torch.Tensor:
        return self.model(image)

    # Using custom or multiple metrics (default_hp_metric=False)
    def on_train_start(self):
        if self.logger:
            self.logger.log_hyperparams(
                {"model": self.hparams, "data": self.dm_hparams},
                {"val/loss": 0, "val/dice_score": 0},
            )

    def training_step(self, batch: dict, batch_idx):
        image = batch["image"]
        output: torch.Tensor = self(image)

        batch_size = image.size(0)

        common_logger_kwargs = {"on_epoch": True, "batch_size": batch_size}

        label = batch.get("label")

        if self.dm_hparams.do_semi:
            progbar_logger_kwargs = {**common_logger_kwargs, "prog_bar": True}

            loss = 0
            if label is not None:
                sup_loss = self.soft_criterion(output, label)
                loss += sup_loss
                self.log("train/sup_loss", sup_loss, **progbar_logger_kwargs)

            channel_dim = 1

            q, q_hat = torch.max(
                torch.softmax(output, dim=channel_dim), dim=channel_dim, keepdim=True
            )

            # Mean prediction for each batch
            q_mean = torch.mean(q.view(q.size(0), -1), dim=channel_dim)

            # Mask per batch
            mask = q_mean >= self.hparams.pseudo_threshold

            if mask.any():
                # Only those images with confident pseudo labels
                masked_image = image[mask]

                # Strong augment image #
                strong_image = torch.stack(tuple(map(self.strong_aug, masked_image)))
                strong_output = self(strong_image)

                # Calculate masked q_hat
                masked_q_hat = q_hat[mask]

                # Measure CE of strong_output with pseudo_labeled weak one
                unsup_loss = self.logit_criterion(strong_output, masked_q_hat)

                self.log("train/unsup_loss", unsup_loss, **progbar_logger_kwargs)

                loss += self.hparams.unsup_weight * unsup_loss

            if loss == 0:
                return None
        else:
            loss = self.soft_criterion(output, label)

        self.log("train/loss", loss, **common_logger_kwargs)

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

        loss = self.soft_criterion(output, label)

        self.log("val/loss", loss, batch_size=1, prog_bar=True)

    def validation_epoch_end(self, outputs):
        # each_score = self.val_dice_metric
        raw_scores = self.val_dice_metric.aggregate()
        self.val_dice_metric.reset()

        self.log_dict(
            {
                f"val/dice_{label}": score
                for label, score in zip(self.labels, raw_scores)
            }
        )

        mean_dice = torch.mean(raw_scores)
        self.log("val/dice_score", mean_dice, prog_bar=True)

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
        # optimizer = torch.optim.SGD(
        #     self.parameters(),
        #     lr=self.hparams.learning_rate,
        #     momentum=self.hparams.momentum,
        #     nesterov=True,
        # )

        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer, T_max=self.trainer.max_epochs
        # )

        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

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
        self.dm_hparams = datamodule.hparams
        if stage is None or stage == "predict":
            self.saver = datamodule.saver
        if stage is None or stage == "fit":
            self.example_input_array = torch.empty(1, 1, *self.roi_size)
            if datamodule.hparams.do_semi:
                self.strong_aug = datamodule.get_strong_aug()
        if stage is None or stage == "fit" or stage == "validate":
            num_labels_with_bg = datamodule.hparams.num_labels_with_bg
            self.post_pred = AsDiscrete(argmax=True, to_onehot=num_labels_with_bg)
            self.post_label = AsDiscrete(to_onehot=num_labels_with_bg)

    def save_scripted(self, path: str):
        torch.jit.script(self.model).save(path)

    def save_model(self, path: str):
        torch.save(self.model.state_dict(), path, pickle_protocol=5)
