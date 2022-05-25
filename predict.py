import os.path

from monai.networks.nets import UNet
from pytorch_lightning import Trainer

from datamodule import DataModule
from segmentor import Segmentor

checkpoint_path = "playground/checkpoints/unet-l6-s2-epoch=20-val_loss=0.49.ckpt"

base_model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=5,
    channels=(2, 4, 8, 16, 32, 64),
    strides=(2, 2, 2, 2, 2),
    num_res_units=3,
)

model = Segmentor.load_from_checkpoint(
    checkpoint_path, model=base_model, sw_batch_size=12, sw_overlap=0.25
)

base_dir = "/mnt/HDD2/flare2022/datasets/AbdomenCT-1K"

dm = DataModule(
    supervised_dir=os.path.join(base_dir, "Subtask1"),
    num_labels_with_bg=5,
    predict_dir=os.path.join(base_dir, "TestImage"),
    roi_size=(128, 128, 32),
    max_workers=4,
)

trainer = Trainer(logger=False, accelerator="gpu", gpus=[0], max_epochs=-1)

preds = trainer.predict(model, datamodule=dm)
