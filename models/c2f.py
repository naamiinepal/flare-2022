import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from monai.transforms import AsDiscrete, KeepLargestConnectedComponent
from torch import Tensor, nn
from monai.data.box_utils import convert_box_mode


from . import BaseModel


class C2FSegmentor(BaseModel):
    def __init__(
        self,
        coarse_model: nn.Module,
        fine_model: nn.Module,
        coarse_weights_path: Optional[str] = None,
        fine_weights_path: Optional[str] = None,
        min_abdomen_size_mm: Optional[Tuple[int, int, int]] = (368, 231, 281),
        **kwargs,
    ):
        self.save_hyperparameters(
            ignore=(
                "coarse_model",
                "fine_model",
                "coarse_weights_path",
                "fine_weights_path",
            )
        )

        super().__init__(**kwargs)

        if coarse_weights_path is not None:
            coarse_model.load_state_dict(
                torch.load(coarse_weights_path, map_location=self.device)
            )
            print(
                "Coarse model weights loaded from: ",
                coarse_weights_path,
            )
        if fine_weights_path is not None:
            fine_model.load_state_dict(
                torch.load(fine_weights_path, map_location=self.device)
            )
            print("Fine model weights loaded from: ", fine_weights_path)

        # Load coarse or fine model
        self.coarse_model = coarse_model.eval()
        self.fine_model = fine_model

        # To get the spacings of images in each batch in forward pass
        self.pix_dims = torch.tensor([])

    def forward(self, image) -> torch.Tensor:
        # pix_dims = self.pix_dims[-self.dm_hparams.batch_size:]
        cropped_images, cropped_indices = self.cropped_image_indices(
            image, self.pix_dims
        )
        scaled_output = []
        if len(cropped_images):
            # (B, 1, H, W, D)
            fine_image = torch.stack(tuple(map(self.resize_fine, cropped_images)))

            # (B, 14, H, W, D)
            output = self.fine_model(fine_image)
            for o, crop_img in zip(output, cropped_images):
                scaled_output.append(
                    F.interpolate(
                        input=o.unsqueeze(0),
                        size=crop_img.shape[1:],
                        mode="trilinear",  # Because output is in (-inf, inf)
                    ).squeeze(0)
                )
        img_shape = image.shape
        final_output = (
            torch.tensor(
                (10000, *((-10000,) * (self.dm_hparams.num_labels_with_bg - 1))),
                dtype=torch.float32,
                device=self.device,
            )
            .expand((img_shape[0], *img_shape[2:], self.dm_hparams.num_labels_with_bg))
            .movedim(-1, 1)
            .clone()  # To have different memory locations
        )
        for i, (scaled_out, ind) in enumerate(zip(scaled_output, cropped_indices)):
            x1, x2, y1, y2, z1, z2 = ind
            final_output[i][:, x1 : x2 + 1, y1 : y2 + 1, z1 : z2 + 1] = scaled_out
        return final_output

    def get_min_sized_abdomen(self, img_pix_dim, boxes_pix: torch.Tensor):
        """
        Returns the bounding box of the minimum sized abdomen in pixels.
        Extracts the physical size of the abdomen from the given bounding box.
        If the abdomen is too small, expands the bounding box to the minimum size,
        else does not change the bounding box.
        This function does not change the original resolution of the image.
        """
        # print(img_pix_dim, "Old box", boxes_pix)
        boxes_mm = torch.concat(
            (boxes_pix[:3] * img_pix_dim, boxes_pix[3:] * img_pix_dim)
        )

        xcenter, ycenter, zcenter, xsize, ysize, zsize = convert_box_mode(
            boxes_mm.unsqueeze(0), src_mode="xyzxyz", dst_mode="cccwhd"
        ).squeeze(0)
        xsize_min, ysize_min, zsize_min = np.maximum(
            self.hparams.min_abdomen_size_mm, (xsize, ysize, zsize)
        )
        xmin, ymin, zmin, xmax, ymax, zmax = (
            convert_box_mode(
                torch.tensor(
                    [[xcenter, ycenter, zcenter, xsize_min, ysize_min, zsize_min]]
                ),
                src_mode="cccwhd",
                dst_mode="xyzxyz",
            )
            .squeeze(0)
            .numpy()
        )

        # convert back to pixel coordinates
        x1 = math.floor(max(0, boxes_pix[0] - abs(xmin)) / img_pix_dim[0])
        y1 = math.floor(max(0, boxes_pix[1] - abs(ymin)) / img_pix_dim[1])
        z1 = math.floor(max(0, boxes_pix[2] - abs(zmin)) / img_pix_dim[2])
        x2 = math.floor(xmax / img_pix_dim[0])
        y2 = math.floor(ymax / img_pix_dim[1])
        z2 = math.floor(zmax / img_pix_dim[2])

        # print("New box: ", x1, y1, z1, x2, y2, z2)
        return x1, y1, z1, x2, y2, z2

    @torch.inference_mode()
    def cropped_image_indices(self, image: Tensor, pix_dims: Tensor):
        # print("Pixdims", pix_dims)
        coarse_image = self.resize_coarse(image)
        coarse_output = self.coarse_model(coarse_image).cpu() >= 0.5
        cropped_images = []
        cropped_indices = []
        has_mask = coarse_output.any()
        if has_mask:
            for c_out, img, pix_dim in zip(coarse_output, image, pix_dims):
                if self.hparams.do_post_process:
                    c_out = self.keep_connected_component_coarse(c_out)
                c_out = c_out.squeeze(0)
                x_indices, y_indices, z_indices = torch.where(c_out)

                # Multiplying less scale rather than all the indices
                x1 = int(x_indices.min() * self.coarse_scale[0])
                x2 = math.ceil(x_indices.max() * self.coarse_scale[0])
                y1 = int(y_indices.min() * self.coarse_scale[1])
                y2 = math.ceil(y_indices.max() * self.coarse_scale[1])
                z1 = int(z_indices.min() * self.coarse_scale[2])
                z2 = math.ceil(z_indices.max() * self.coarse_scale[2])
                # print("Here")
                # x1, y1, z1, x2, y2, z2 = self.get_min_sized_abdomen(
                #     img_pix_dim=torch.tensor(pix_dim).cpu(),
                #     boxes_pix=torch.tensor([x1, y1, z1, x2, y2, z2]).cpu(),
                # )
                cropped_indices.append((x1, x2, y1, y2, z1, z2))
                cropped_images.append(img[:, x1 : x2 + 1, y1 : y2 + 1, z1 : z2 + 1])
        return cropped_images, cropped_indices

    def get_fine_image(self, image):
        cropped_images, _ = self.cropped_image_indices(image)
        fine_image = torch.stack(tuple(map(self.resize_fine, cropped_images)))
        return fine_image

    def training_step(self, batch: dict, batch_idx):
        image = batch["image"]
        # print(batch['image_meta_dict']['pixdim'])
        self.pix_dims = batch["image_meta_dict"]["pixdim"][:, 1:4]
        # print("Pix dim", self.pix_dims)

        common_logger_kwargs = {
            "on_epoch": True,
            "batch_size": image.size(0),
        }

        label = batch.get("label")

        if self.dm_hparams.do_semi:
            progbar_logger_kwargs = {**common_logger_kwargs, "prog_bar": True}

            # print("Before Sup")

            sup_loss = 0.0
            if label is not None:
                output = self.fine_model(image)
                sup_loss = self.criterion(output, label)
            else:
                # Image is not forefround cropped, so need to crop it
                output = self(image)

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
                strong_output = self.fine_model(strong_image)

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
            output = self.fine_model(image)
            loss = self.criterion(output, label)

        self.log("train/loss", loss, **common_logger_kwargs)

        if loss == 0:
            return None
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        label = batch["label"]
        image = batch["image"]
        self.pix_dims = batch["image_meta_dict"]["pixdim"][:, 1:4]

        output = self.sliding_inferer(image, self)

        if not self.trainer.fast_dev_run and self.logger is not None and batch_idx == 0:
            self.plot_image(torch.argmax(output, dim=1, keepdim=True), tag="pred")

            # Plot label only once
            if not self.is_first_plot:
                self.plot_image(label, tag="label")
                self.is_first_plot = True

        self.compute_dice_score(output, label)

        loss = self.criterion(output, label)

        self.log("val/loss", loss, batch_size=1, prog_bar=True)

        # # NOTE: this is a temporary workaround to view validation output, delete later
        # batch_meta_data = {
        #     k: v.cpu() if isinstance(v, torch.Tensor) else v
        #     for k, v in batch["image_meta_dict"].items()
        # }

        # for i, out in enumerate(output):
        #     if self.hparams.do_post_process:
        #         out = self.keep_connected_component(self.post_pred(out))
        #     argmax_out = out.argmax(dim=0)
        #     meta_data = {k: v[i] for k, v in batch_meta_data.items()}
        #     self.saver(argmax_out, meta_data)

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        image = batch["image"]
        self.pix_dims = batch["image_meta_dict"]["pixdim"][:, 1:4]

        output = self.sliding_inferer(image, self)

        batch_meta_data = {
            k: v.cpu() if isinstance(v, torch.Tensor) else v
            for k, v in batch["image_meta_dict"].items()
        }

        for i, out in enumerate(output):
            if self.hparams.do_post_process:
                out = self.keep_connected_component(self.post_pred(out))
            argmax_out = out.argmax(dim=0, keepdim=True)
            meta_data = {k: v[i] for k, v in batch_meta_data.items()}
            self.saver(argmax_out, meta_data)

    def resize_coarse(self, img_batch: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            input=img_batch,
            size=self.dm_hparams.coarse_roi_size,
            mode="trilinear",
        )

    def resize_fine(self, image: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            input=image.unsqueeze(0),
            size=self.dm_hparams.fine_roi_size,
            mode="trilinear",
        ).squeeze(0)

    def setup(self, stage: Optional[str] = None):
        super().setup(stage)

        dm_hparams = self.dm_hparams

        num_labels_with_bg: int = dm_hparams.num_labels_with_bg

        self.post_pred = AsDiscrete(argmax=True, to_onehot=num_labels_with_bg)
        self.post_label = AsDiscrete(to_onehot=num_labels_with_bg)
        self.coarse_scale = np.array(dm_hparams.intermediate_roi_size) / np.array(
            dm_hparams.coarse_roi_size
        )

        if stage is None or stage == "fit" or stage == "validate":
            self.image_scaler = 255 / (num_labels_with_bg - 1)

        self.keep_connected_component_coarse = KeepLargestConnectedComponent(
            independent=False,
            connectivity=self.hparams.connectivity,
        )

        self.keep_connected_component = KeepLargestConnectedComponent(
            applied_labels=range(1, num_labels_with_bg),
            is_onehot=False,
            independent=True,
            connectivity=self.hparams.connectivity,
        )

    def save_scripted(self, path: str, use_gpu: bool = True):
        device = "cuda" if use_gpu else "cpu"
        fine_model = self.fine_model.eval().to(device)
        coarse_model = self.coarse_model.eval().to(device)

        torch.jit.script(fine_model).save(f"fine_{path}")
        torch.jit.script(coarse_model).save(f"coarse_{path}")

    def save_model(self, path: str):
        torch.save(self.fine_model.state_dict(), f"fine_{path}", pickle_protocol=5)
        torch.save(self.coarse_model.state_dict(), f"coarse_{path}", pickle_protocol=5)
