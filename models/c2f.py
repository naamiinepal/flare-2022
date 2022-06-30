from typing import Optional

import torch
import torch.nn.functional as F
from monai.transforms import AsDiscrete, CropForeground, KeepLargestConnectedComponent
from torch import Tensor, nn

from custom_transforms import CustomResize
from models.basemodel import BaseModel


class C2FSegmentor(BaseModel):
    def __init__(
        self,
        coarse_model: nn.Module,
        fine_model: nn.Module,
        coarse_weights_path: Optional[str] = None,
        fine_weights_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if coarse_weights_path is not None:
            coarse_model.load_state_dict(torch.load(coarse_weights_path))
            print(
                "Coarse model weights loaded from: ",
                coarse_weights_path,
            )
        if fine_weights_path is not None:
            fine_model.load_state_dict(torch.load(fine_weights_path))
            print("Fine model weights loaded from: ", fine_weights_path)

        # Load coarse or fine model
        self.coarse_model = coarse_model
        self.fine_model = fine_model

    def forward(self, image) -> torch.Tensor:
        cropped_images, cropped_indices = self.cropped_image_indices(image)
        fine_image = torch.stack(tuple(map(self.resize_fine, cropped_images)))
        output = self.fine_model(fine_image)
        scaled_output = []
        for o, img in zip(output, cropped_images):
            scaled_output.append(
                F.interpolate(
                    input=o.unsqueeze(0),
                    size=img.shape,
                    mode="nearest",
                ).squeeze(0)
            )
        final_output = torch.zeros_like(image)
        for i, (scaled_out, ind) in enumerate(zip(scaled_output, cropped_indices)):
            x1, x2, y1, y2, z1, z2 = ind
            final_output[i][x1 : x2 + 1, y1 : y2 + 1, z1 : z2 + 1] = scaled_out
        return final_output

    def cropped_image_indices(self, image: Tensor):
        coarse_image = torch.stack(tuple(map(self.resize_coarse, image)))
        coarse_output = self.coarse_model(coarse_image).squeeze(1)
        cropped_images = []
        cropped_indices = []
        for c_out, img in zip(coarse_output, image):
            coarse_scale = torch.astensor(img.shape) / self.dm_hparams.coarse_roi_size
            indices = torch.vstack(torch.where(c_out >= 0.5)) * coarse_scale
            x1 = int(indices[0].min())
            x2 = int(indices[0].max().ceil())
            y1 = int(indices[1].min())
            y2 = int(indices[1].max().ceil())
            z1 = int(indices[2].min())
            z2 = int(indices[2].max().ceil())
            cropped_indices.append((x1, x2, y1, y2, z1, z2))
            cropped_images.append(img[x1 : x2 + 1, y1 : y2 + 1, z1 : z2 + 1])

        return cropped_images, cropped_indices

    def get_fine_image(self, image):
        cropped_images, _ = self.cropped_image_indices(image)
        fine_image = torch.stack(tuple(map(self.resize_fine, cropped_images)))
        return fine_image

    def training_step(self, batch: dict, batch_idx):
        image = batch["image"]

        output: torch.Tensor = self.fine_model(image)

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
            else:
                # Image is not forefround cropped, so need to crop it
                with torch.inference_mode():
                    fine_image = self.get_fine_image(image)

                output = self.fine_model(fine_image)

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
        self.crop_foreground = CropForeground()
        self.resize_coarse = CustomResize(
            roi_size=self.dm_hparams.coarse_roi_size, mode="trilinear"
        )

        self.resize_fine = CustomResize(
            roi_size=self.dm_hparams.fine_roi_size, mode="trilinear"
        )

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
