from typing import Iterable, Literal, Optional

import pytorch_lightning as pl
import torch
from torch import nn
from monai.inferers import SlidingWindowInferer
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.visualize.img2tensorboard import add_animated_gif

from datamodules import BaseDataModule


class BaseModel(pl.LightningModule):

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
        connectivity: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()

    # Using custom or multiple metrics (default_hp_metric=False)
    def on_train_start(self):
        if self.logger:
            self.logger.log_hyperparams(
                {"model": self.hparams, "data": self.dm_hparams},
                {"val/loss": 0, "val/dice_score": 0},
            )

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        label = batch["label"]
        image = batch["image"]

        output = self.sliding_inferer(image, self)

        if self.logger is not None and batch_idx == 0:
            self.plot_image(torch.argmax(output, dim=1, keepdim=True), tag="pred")

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
            if self.hparams.do_post_process:
                out = self.keep_connected_component(self.post_pred(out))
            argmax_out = out.argmax(dim=0, keepdim=True)
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
        datamodule: BaseDataModule = self.trainer.datamodule
        self.roi_size = datamodule.hparams.roi_size
        self.dm_hparams = datamodule.hparams

        self.sliding_inferer = SlidingWindowInferer(
            self.roi_size,
            self.hparams.sw_batch_size,
            self.hparams.sw_overlap,
            mode=self.hparams.sw_mode,
            cache_roi_weight_map=True,
            # device="cpu",
        )

        if stage is None or stage == "predict":
            self.saver = datamodule.saver
        if stage is None or stage == "fit":
            self.example_input_array = torch.empty(1, 1, *self.roi_size)
            if datamodule.hparams.do_semi:
                self.strong_aug = datamodule.get_strong_aug()
        if stage is None or stage == "fit" or stage == "validate":
            self.is_first_plot = False


class SingleBaseModel(BaseModel):
    """
    Models with single backbone inside
    """

    def __init__(
        self,
        model: nn.Module,
        model_weights_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.save_hyperparameters(ignore=("model", "model_weights_path"))

        if model_weights_path is not None:
            print("Model weights loaded from:", model_weights_path)
            model.load_state_dict(torch.load(model_weights_path))

        self.model = model

    def forward(self, image) -> torch.Tensor:
        return self.model(image)

    def save_scripted(self, path: str, use_gpu: bool = True):
        device = "cuda" if use_gpu else "cpu"
        model = self.model.eval().to(device)
        torch.jit.script(model).save(path)

    def save_model(self, path: str):
        torch.save(self.model.state_dict(), path, pickle_protocol=5)
