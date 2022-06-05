from monai.networks.nets import UNet

from segmentor import Segmentor

base_model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=14,
    channels=(8, 16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2, 2),
)

checkpoint_path = (
    "checkpoints/unet-l6-s8-256-newloss-dataaug-epoch=55-val/loss=0.49.ckpt"
)

model = Segmentor.load_from_checkpoint(checkpoint_path, model=base_model)

model.save_scripted("flare_supervised_checkpoint.pt")
