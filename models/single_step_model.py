from typing import Optional

import torch
from monai.transforms import AsDiscrete, KeepLargestConnectedComponent

from models.basemodel import SingleBaseModel


class SingleStepModel(SingleBaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.save_hyperparameters()

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

            channel_dim = 1

            # q_hat = torch.argmax(output, dim=channel_dim, keepdim=True)
            q, q_hat = torch.max(
                torch.softmax(output, dim=channel_dim), dim=channel_dim, keepdim=True
            )

            # Mean prediction for each batch
            q_thres = torch.mean(q.view(q.size(0), -1), dim=channel_dim)

            # sm_fore = torch.amax(
            #     torch.softmax(output, dim=channel_dim)[:, 1:, ...], dim=channel_dim
            # )
            # q_thres = torch.mean(sm_fore.view(sm_fore.size(0), -1), dim=channel_dim)

            # mask = N
            # Mask per batch
            mask = q_thres >= self.hparams.pseudo_threshold

            unsup_loss = 0.0

            if mask.any():
                # Only those images with confident pseudo labels
                masked_image = image[mask]

                # Calculate masked q_hat
                masked_q_hat = q_hat[mask]

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

    def setup(self, stage: Optional[str] = None):
        super().setup(stage)

        num_labels_with_bg: int = self.trainer.datamodule.hparams.num_labels_with_bg

        self.post_pred = AsDiscrete(argmax=True, to_onehot=num_labels_with_bg)
        self.post_label = AsDiscrete(to_onehot=num_labels_with_bg)

        if stage is None or stage == "fit" or stage == "validate":
            self.image_scaler = 255 / (num_labels_with_bg - 1)

        self.keep_connected_component = KeepLargestConnectedComponent(
            applied_labels=range(1, num_labels_with_bg),
            is_onehot=True,
            independent=True,
            connectivity=self.hparams.connectivity,
        )
