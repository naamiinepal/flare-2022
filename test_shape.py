# import sys
import os.path

# from segmentor import Segmentor
# from monai.networks.nets import UNet
import sys

import torch
from tqdm import tqdm

from datamodule import DataModule

#
# import torch


# from tqdm import tqdm


BASE_DIR = "/mnt/HDD2/flare2022/datasets/FLARE2022"

dm = DataModule(
    supervised_dir=os.path.join(BASE_DIR, "Training/FLARE22_LabeledCase50/"),
    predict_dir=os.path.join(BASE_DIR, "Validation/"),
    val_ratio=0.001,
    num_labels_with_bg=14,
    ds_cache_type=None,
    batch_size=1,
    max_workers=4,
    roi_size=(128, 128, 64),
)

dm.setup("fit")


dl = dm.train_dataloader()

# segmentor = Segmentor(
#     model=UNet(
#         spatial_dims=3,
#         in_channels=1,
#         out_channels=dm.hparams.num_labels_with_bg,
#         channels=(4, 8, 16, 32, 64, 128),
#         strides=(2, 2, 2, 2, 2),
#         num_res_units=3,
#         norm="batch",
#         bias=False,
#     )
# ).eval()

# with torch.inference_mode():
# min_value = sys.maxsize
# max_value = 0
# for batch in tqdm(dl):
#     image = batch["image"]
#     min_value = min(min_value, image.min())
#     max_value = max(max_value, image.max())
#     # label = batch["label"]
#     # print(image.shape, label.shape)
#     # pred = segmentor(batch["image"])
#     # print(pred.shape)

# print(min_value, max_value)

min_shape = torch.tensor((sys.maxsize, sys.maxsize, sys.maxsize))
max_shape = torch.zeros(3, dtype=int)

for batch in tqdm(dl):
    image_shape = torch.as_tensor(batch["image"].shape)[-3:]
    min_shape = torch.min(min_shape, image_shape)
    max_shape = torch.max(max_shape, image_shape)

print(min_shape, max_shape)
