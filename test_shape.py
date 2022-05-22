# import sys
#
import torch
from monai.networks.nets import UNet

from datamodule import DataModule
from model import Segmentor

# from tqdm import tqdm


dm = DataModule(
    supervised_dir="/mnt/HDD2/flare2022/datasets/AbdomenCT-1K/Subtask1",
    val_ratio=0.1,
    num_labels_with_bg=5,
    ds_cache_type=None,
    batch_size=1,
    max_workers=1,
    roi_size=(256, 256, 32),
)

dm.setup()

dl = dm.train_dataloader()

segmentor = Segmentor(
    model=UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=dm.hparams.num_labels_with_bg,
        channels=[4, 8, 16, 32, 64, 128],
        strides=[2, 2, 2, 2, 2],
        num_res_units=3,
        norm="batch",
        bias=False,
    )
).eval()

with torch.inference_mode():
    for batch in dl:
        print(batch["image"].shape)
        print(batch["label"].shape)
        print(segmentor(batch["image"]).shape)
        break

# min_shape = torch.tensor((sys.maxsize, sys.maxsize, sys.maxsize))
# max_shape = torch.zeros(3, dtype=int)

# for batch in tqdm(dl):
#     image_shape = torch.as_tensor(batch["image"].shape)[-3:]
#     min_shape = torch.min(min_shape, image_shape)
#     max_shape = torch.max(max_shape, image_shape)

# print(min_shape, max_shape)
