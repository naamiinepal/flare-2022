from monai.networks.nets import UNet
from models.coarse_model import CoarseModel

base_model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    channels=(4, 8, 16, 32, 64),
    strides=(2, 2, 2, 2),
    act="relu",
)

checkpoint_path = (
    "checkpoints/c2f/c2f-fine-unet-l5-s16-256-new//epoch=47-val/loss=0.35-v1.ckpt"
)

model = CoarseModel.load_from_checkpoint(
    checkpoint_path, model=base_model, model_weights_path=None
)

model.save_model("coarse_unscripted.pt")
