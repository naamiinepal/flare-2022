from typing import Optional

import torch
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import KeepLargestConnectedComponent

from models.basemodel import SingleBaseModel


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
        super.__init(pseudo_threshold=pseudo_threshold, **kwargs)

        # Captured by the parent class
        self.save_hyperparameters(ignore="pseudo_threshold")

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

    def discretize_output(self, output: torch.Tensor) -> torch.Tensor:
        return output >= self.hparams.output_threshold

    def setup(self, stage: Optional[str] = None):
        super().setup(stage=stage)

        # onehot and independent can be anything, but optimized for implementation
        self.keep_connected_component = KeepLargestConnectedComponent(
            applied_labels=1,
            is_onehot=True,
            independent=False,
            connectivity=self.hparams.connectivity,
        )
