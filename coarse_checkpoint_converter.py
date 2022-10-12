from monai.networks.nets import UNet
from models.coarse_model import CoarseModel

base_model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    kernel_size=(5, 5, 3),
    up_kernel_size=(5, 5, 3),
    channels=(16, 16, 32, 32, 64),
    strides=(2, 2, 2, 2),
    act="relu",
)

checkpoint_path = "checkpoints/coarse/unet-l5-s16-64-customresize-semi-kernel-553/epoch=11-val/loss=0.1016.ckpt"

model = CoarseModel.load_from_checkpoint(
    checkpoint_path, model=base_model, model_weights_path=None
)

model.save_model("coarse_boundingmask_semi_unscripted.pt")
# model.save_scripted("coarse_boundingmask_semi.ts")
