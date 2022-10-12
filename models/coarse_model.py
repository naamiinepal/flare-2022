from typing import Iterable, Optional

import torch
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import KeepLargestConnectedComponent

from . import SingleBaseModel


class CoarseModel(SingleBaseModel):

    val_dice_metric = DiceMetric()

    criterion = DiceCELoss(sigmoid=True, lambda_ce=0.25)

    def __init__(
        self, output_threshold: float = 0.5, pseudo_threshold: float = 0.4, **kwargs
    ):
        """
        Psuedo_threshold is redefined here to change the defaults
        The pseudo_threshold is 0.5 less than the one used in other models
        Because this model is binary
        """
        super().__init__(pseudo_threshold=pseudo_threshold, **kwargs)

    def training_step(self, batch: dict, batch_idx):
        image = batch[0]
        output: torch.Tensor = self(image)

        common_logger_kwargs = {
            "on_epoch": True,
            "batch_size": image.size(0),
        }

        label = batch[1]  # NOTE: Change this for unlabeled later

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

    def discretize_output(self, output: torch.Tensor) -> torch.Tensor:
        return output >= self.hparams.output_threshold

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        label = batch["label"]
        image = batch["image"]

        output = self.sliding_inferer(image, self)

        if not self.trainer.fast_dev_run and self.logger is not None and batch_idx == 0:
            self.plot_image(self.discretize_output(output), tag="pred")

            # Plot label only once
            if not self.is_first_plot:
                self.plot_image(label, tag="label")
                self.is_first_plot = True

        self.compute_dice_score(output, label)

        loss = self.criterion(output, label)

        self.log("val/loss", loss, batch_size=1, prog_bar=True)

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
            pred_out = self.discretize_output(out)
            if self.hparams.do_post_process:
                pred_out = self.keep_connected_component(pred_out)
            meta_data = {k: v[i] for k, v in batch_meta_data.items()}
            self.saver(pred_out, meta_data)

    def compute_dice_score(
        self, output: Iterable[torch.Tensor], label: Iterable[torch.Tensor]
    ):
        post_output = self.discretize_output(output.squeeze(0))
        if self.hparams.do_post_process:
            # old_output = post_output.clone()
            post_output = self.keep_connected_component(post_output)
            # print("Diff: ", (post_output ^ old_output).sum())

        self.val_dice_metric(post_output.unsqueeze(0), label)

    def setup(self, stage: Optional[str] = None):
        super().setup(stage=stage)

        # onehot and independent can be anything, but optimized for implementation
        self.keep_connected_component = KeepLargestConnectedComponent(
            independent=False,
            connectivity=self.hparams.connectivity,
        )

        self.image_scaler = 255
