from typing import Iterable, Literal, Optional

import pytorch_lightning as pl
import torch
from monai.inferers import SlidingWindowInferer
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete, KeepLargestConnectedComponent, CropForeground
from monai.visualize.img2tensorboard import add_animated_gif
from torch import Tensor, nn

from datamodules.datamodule import DataModule


class C2FSegmentor(pl.LightningModule):

    val_dice_metric = DiceMetric(include_background=False, reduction="mean_batch")

    criterion = DiceCELoss(
        include_background=False, to_onehot_y=True, softmax=True, lambda_ce=0.25
    )

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
        coarse_model: nn.Module,
        fine_model: nn.Module,
        pseudo_threshold: float = 0.9,
        unsup_weight: float = 1,
        learning_rate: float = 0.03,
        sw_batch_size: int = 4,
        sw_overlap: float = 0.1,
        sw_mode: Literal["constant", "gaussian"] = "gaussian",
        plateu_patience: int = 2,
        plateu_factor: float = 0.1,
        momentum: float = 0.9,
        monitor: str = "val/loss",
        do_post_process: bool = True,
        is_coarse: bool = True,
        connectivity: Optional[int] = None,
        coarse_model_weights_path: Optional[str] = None,
        fine_model_weights_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["coarse_model", "fine_model"])

        # if model_weights_path is not None:
        #     print("Model weights loaded from:", model_weights_path)
        #     model.load_state_dict(torch.load(model_weights_path))

        if self.hparams.coarse_model_weights_path is not None:
            coarse_model.load_state_dict(
                torch.load(self.hparams.coarse_model_weights_path)
            )
            print(
                "Coarse model weights loaded from:",
                self.hparams.coarse_model_weights_path,
            )

        if self.hparams.fine_model_weights_path is not None:
            fine_model.load_state_dict(torch.load(self.hparams.fine_model_weights_path))
            print(
                "Fine model weights loaded from:", self.hparams.fine_model_weights_path
            )
        # Load coarse or fine model
        self.coarse_model = coarse_model
        self.fine_model = fine_model
        if self.hparams.is_coarse:
            self.model = self.coarse_model
        else:
            self.model = self.fine_model

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

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        label = batch["label"]
        image = batch["image"]

        output = self.sliding_inferer(image, self)

        if self.logger is not None and batch_idx == 0:
            pred_output_mask = torch.argmax(output, dim=1, keepdim=True)
            self.plot_image(pred_output_mask, tag="pred")
            if self.hparams.is_coarse:
                self.plot_image(self.crop_foreground(pred_output_mask), tag="pred_bbox")

            # Plot label only once
            if not self.is_first_plot:
                self.plot_image(label, tag="label")
                if self.hparams.is_coarse:
                    self.plot_image(self.crop_foreground(label), tag="label_bbox")
                self.is_first_plot = True

        self.compute_dice_score(output, label)

        loss = self.criterion(output, label)

        self.log("val/loss", loss, batch_size=1, prog_bar=True)

    def plot_image(self, image: Tensor, tag: str):
        add_animated_gif(
            self.logger.experiment,
            f"{tag}_HWD",
            image[0].cpu().numpy(),
            max_out=1,
            frame_dim=-1,
            scale_factor=self.image_scaler,
            global_step=self.global_step,
        )

    def validation_epoch_end(self, outputs):
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

        output = self.sliding_inferer(image, self)

        batch_meta_data = {
            k: v.cpu() if isinstance(v, torch.Tensor) else v
            for k, v in batch["image_meta_dict"].items()
        }

        for i, out in enumerate(output):
            argmax_out = out.argmax(dim=0)
            meta_data = {k: v[i] for k, v in batch_meta_data.items()}
            self.saver(argmax_out, meta_data)

    def compute_dice_score(
        self, output: Iterable[torch.Tensor], label: Iterable[torch.Tensor]
    ):
        post_output = self.post_pred(output.squeeze(0))
        if self.hparams.do_post_process:
            post_output = self.keep_connected_component(post_output)

        post_label = self.post_label(label.squeeze(0)).unsqueeze(0)

        self.val_dice_metric(post_output.unsqueeze(0), post_label)

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
        self.roi_size = (
            datamodule.hparams.coarse_roi_size
            if self.hparams.is_coarse
            else datamodule.hparams.fine_roi_size
        )
        self.dm_hparams = datamodule.hparams

        self.sliding_inferer = SlidingWindowInferer(
            self.roi_size,
            self.hparams.sw_batch_size,
            self.hparams.sw_overlap,
            mode=self.hparams.sw_mode,
            cache_roi_weight_map=True,
        )

        num_labels_with_bg: int = datamodule.hparams.num_labels_with_bg

        self.post_pred = AsDiscrete(argmax=True, to_onehot=num_labels_with_bg)
        self.post_label = AsDiscrete(to_onehot=num_labels_with_bg)
        self.keep_connected_component = KeepLargestConnectedComponent(
            applied_labels=range(1, num_labels_with_bg),
            is_onehot=True,
            independent=True,
            connectivity=self.hparams.connectivity,
        )
        self.crop_foreground = CropForeground()

        if stage is None or stage == "predict":
            self.saver = datamodule.saver
            # TODO: Coarse Model --> Fine Model
        if stage is None or stage == "fit":
            self.example_input_array = torch.empty(1, 1, *self.roi_size)
            if datamodule.hparams.do_semi:
                self.strong_aug = datamodule.get_strong_aug()
        if stage is None or stage == "fit" or stage == "validate":
            self.image_scaler = 255 / (num_labels_with_bg - 1)
            self.is_first_plot = False

    def save_scripted(self, path: str):
        torch.jit.script(self.model).save(path)

    def save_model(self, path: str):
        torch.save(self.model.state_dict(), path, pickle_protocol=5)
