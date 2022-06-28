from monai.networks.nets import UNet

from segmentor import Segmentor

base_model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=14,
    channels=(4, 8, 16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2, 2, 2),
    num_res_units=3,
    act="relu",
)

# "checkpoints/unet-l7-s4-256-spacing-res2-weak-aug/epoch=77-val/loss=0.49.ckpt"
checkpoint_path = "checkpoints/unet-l7-s4-256-customresize-res3-semi-diCE0.25/epoch=129-val/loss=0.45-v1.ckpt"

model = Segmentor.load_from_checkpoint(
    checkpoint_path, model=base_model, model_weights_path=None
)

# model.save_scripted("flare_supervised_checkpoint.pt")
model.save_model("flare_supervised_unscripted.pt")
