from monai.networks.nets import UNet

from segmentor import Segmentor

base_model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=5,
    channels=(4, 8, 16, 32, 64, 128),
    strides=(2, 2, 2, 2, 2),
)

checkpoint_path = "playground/checkpoints/unet-l6-s4-r0-epoch=41-val_loss=0.29.ckpt"

model = Segmentor.load_from_checkpoint(checkpoint_path, model=base_model)

model.save_scripted("abdomen_checkpoint.pt")
