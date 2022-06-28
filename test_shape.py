# import sys
import os.path
import numpy as np
from scipy.stats import mode

# from segmentor import Segmentor
# from monai.networks.nets import UNet
from tqdm import tqdm

from datamodules.c2f_datamodule import C2FDataModule

#
# import torch


# from tqdm import tqdm


BASE_DIR = "/mnt/HDD2/flare2022/datasets/FLARE2022"

dm = C2FDataModule(
    supervised_dir=os.path.join(BASE_DIR, "Training/FLARE22_LabeledCase50/"),
    predict_dir=os.path.join(BASE_DIR, "Training/FLARE22_LabeledCase50/images"),
    val_ratio=0.001,
    num_labels_with_bg=14,
    ds_cache_type=None,
    batch_size=1,
    max_workers=4,
    # roi_size=(128, 128, 64),
    # pixdim=(3, 3, 2),
)

print(dm.hparams.supervised_dir)

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


def print_stats(image_shapes):
    shape_array = np.array(image_shapes)
    print("Len of image_shapes:", len(image_shapes))
    print("Min shape:", shape_array.min(axis=0))
    print("Max shape:", shape_array.max(axis=0))
    print("Mean shape:", shape_array.mean(axis=0))
    print("Std shape:", shape_array.std(axis=0))
    print("Median shape:", np.median(shape_array, axis=0))
    print("Mode shape:", mode(shape_array, axis=0), end="\n\n")


image_shapes = []

for i, batch in enumerate(tqdm(dl), start=1):
    image_shape = batch["image"].shape[-3:]
    image_shapes.append(image_shape)

    if i % 100 == 0:
        print(f"Stats of first {i} images:")
        print_stats(image_shapes)

print("Final stats:")
print_stats(image_shapes)
