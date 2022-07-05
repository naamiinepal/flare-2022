import torch_tensorrt
from monai.networks.nets import UNet

from models.c2f import C2FSegmentor

# coarse_checkpoint = "flare_coarse_model.pt"
fine_checkpoint_path = (
    "checkpoints/c2f/c2f-fine-unet-l5-s16-256-new/epoch=48-val/loss=0.35.ckpt"
)

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

model = C2FSegmentor.load_from_checkpoint(
    checkpoint_path=fine_checkpoint_path,
    coarse_model=coarse_model,
    fine_model=fine_model,
)

# model.save_model("flare_model.pt")
model.save_scripted("flare_model.ts", use_tensorrt=True)
