import pytorch_lightning as pl
from monai.networks.nets import UNet

from datamodules.c2f_datamodule import C2FDataModule
from models.c2f import C2FSegmentor

pl.seed_everything(42)

coarse_model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    channels=(4, 8, 16, 32, 64),
    strides=(2, 2, 2, 2),
    act="relu",
)

fine_model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=14,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    act="relu",
)

model = C2FSegmentor(coarse_model=coarse_model, fine_model=fine_model)

dm = C2FDataModule(
    supervised_dir="/mnt/HDD2/flare2022/datasets/FLARE2022/Training/FLARE22_LabeledCase50",
    val_ratio=0.05,
    coarse_roi_size=(128, 128, 64),
    fine_roi_size=(192, 192, 96),
    intermediate_roi_size=(256, 256, 128),
    pin_memory=False,
)

trainer = pl.Trainer(logger=True, accelerator="cpu", fast_dev_run=True, max_epochs=1)

trainer.validate(model, datamodule=dm)

model.to_torchscript("c2f_scripted.pt")
