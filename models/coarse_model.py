from typing import Iterable, Literal, Optional

import pytorch_lightning as pl
import torch
from monai.inferers import SlidingWindowInferer
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import KeepLargestConnectedComponent
from monai.visualize.img2tensorboard import add_animated_gif
from torch import nn

from datamodules.datamodule import DataModule


class CoarseModel(pl.LightningModule):

    val_dice_metric = DiceMetric()

    criterion = DiceCELoss(sigmoid=True, lambda_ce=0.25)

    def __init__(
        self,
        model: nn.Module,
        model_weights_path: Optional[str] = None,
        output_threshold: float = 0.5,
        pseudo_threshold: float = 0.9,
        unsup_weight: float = 1,
        learning_rate: float = 0.03,
        sw_batch_size: int = 4,
        sw_overlap: float = 0.1,
        sw_mode: Literal["constant", "gaussian"] = "gaussian",
        plateu_patience: int = 2,
        plateu_factor: float = 0.1,
        monitor: str = "val/loss",
        do_post_process: bool = True,
        connectivity: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["model", "model_weights_path"])

        if model_weights_path is not None:
            print("Model weights loaded from:", model_weights_path)
            model.load_state_dict(torch.load(model_weights_path))

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

        common_logger_kwargs = {
            "on_epoch": True,
            "batch_size": image.size(0),
        }

        label = batch.get("label")

        if self.dm_hparams.do_semi:
            progbar_logger_kwargs = {**common_logger_kwargs, "prog_bar": True}

            sup_loss = 0.0
            if label is not None:
                sup_loss = self.criterion(output, label)

            # Need to calculate distance from the mean for binary output
            abs_mean_deviation = torch.abs(torch.sigmoid(output) - 0.5)

            # Mean prediction for each batch
            q_thres = torch.mean(
                abs_mean_deviation.view(abs_mean_deviation.size(0), -1), dim=1
            )

            # Mask per batch
            mask = q_thres >= self.hparams.pseudo_threshold

            unsup_loss = 0.0

            if mask.any():
                # Only those images with confident pseudo labels
                masked_image = image[mask]

                # Calculate masked q_hat
                masked_q_hat = self.discretize_output(output[mask])

                # Strong augment image #
                strong_image = torch.stack(tuple(map(self.strong_aug, masked_image)))
                strong_output = self(strong_image)

                # Measure Dice of strong_output with augmented pseudo_labeled weak one
                unsup_loss = self.criterion(strong_output, masked_q_hat)

            self.log(
                "train/unsup_count",
                mask.sum(dtype=torch.float32),
                on_epoch=True,
                reduce_fx="sum",
            )

            self.log("train/unsup_loss", unsup_loss, **progbar_logger_kwargs)
            self.log("train/sup_loss", sup_loss, **progbar_logger_kwargs)

            loss = sup_loss + self.hparams.unsup_weight * unsup_loss
        else:
            loss = self.criterion(output, label)

        self.log("train/loss", loss, **common_logger_kwargs)

        if loss == 0:
            return None
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        label = batch["label"]
        image = batch["image"]

        output = self.sliding_inferer(image, self)

        if self.logger is not None and batch_idx == 0:
            self.plot_image(self.discretize_output(output), tag="pred")

            # Plot label only once
            if not self.is_first_plot:
                self.plot_image(label, tag="label")
                self.is_first_plot = True

        self.compute_dice_score(output, label)

        loss = self.criterion(output, label)

        self.log("val/loss", loss, batch_size=1, prog_bar=True)

    def plot_image(self, image: torch.Tensor, tag: str):
        add_animated_gif(
            self.logger.experiment,
            f"{tag}_HWD",
            image[0].cpu().numpy(),
            max_out=1,
            frame_dim=-1,
            scale_factor=255,
            global_step=self.global_step,
        )

    def validation_epoch_end(self, outputs):
        dice_score = self.val_dice_metric.aggregate()
        self.val_dice_metric.reset()

        self.log("val/dice_score", dice_score, prog_bar=True)

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        image = batch["image"]

        output = self.sliding_inferer(image, self)

        batch_meta_data = {
            k: v.cpu() if isinstance(v, torch.Tensor) else v
            for k, v in batch["image_meta_dict"].items()
        }

        for i, out in enumerate(output):
            pred_out = self.descretize_output(out)
            meta_data = {k: v[i] for k, v in batch_meta_data.items()}
            self.saver(pred_out, meta_data)

    def compute_dice_score(
        self, output: Iterable[torch.Tensor], label: Iterable[torch.Tensor]
    ):
        post_output = self.discretize_output(output)
        if self.hparams.do_post_process:
            post_output = self.keep_connected_component(
                post_output.squeeze(0)
            ).unsqueeze(0)

        self.val_dice_metric(post_output, label)

    def discretize_output(self, output: torch.Tensor) -> torch.Tensor:
        return output >= self.hparams.output_threshold

    def configure_optimizers(self):
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

        self.sliding_inferer = SlidingWindowInferer(
            self.roi_size,
            self.hparams.sw_batch_size,
            self.hparams.sw_overlap,
            mode=self.hparams.sw_mode,
            cache_roi_weight_map=True,
        )

        # onehot and independent can be anything, but optimized for implementation
        self.keep_connected_component = KeepLargestConnectedComponent(
            applied_labels=1,
            is_onehot=True,
            independent=False,
            connectivity=self.hparams.connectivity,
        )

        if stage is None or stage == "predict":
            self.saver = datamodule.saver
        if stage is None or stage == "fit":
            self.example_input_array = torch.empty(1, 1, *self.roi_size)
            if datamodule.hparams.do_semi:
                self.strong_aug = datamodule.get_strong_aug()
        if stage is None or stage == "fit" or stage == "validate":
            self.is_first_plot = False

    def save_scripted(self, path: str):
        torch.jit.script(self.model).save(path)

    def save_model(self, path: str):
        torch.save(self.model.state_dict(), path, pickle_protocol=5)
