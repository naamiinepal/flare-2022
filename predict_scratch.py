import gc
import os.path
from argparse import ArgumentParser, Namespace
from glob import glob

import numpy as np
import torch
from monai.data import DataLoader, Dataset
from monai.inferers import SlidingWindowInferer
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    ToTensord,
)

from custom_transforms import CustomResized
from saver import NiftiSaver

# from saver import NiftiSaver


num_labels_with_bg = 14
roi_size = (128, 128, 64)


def main(params: Namespace):
    device = torch.device("cpu" if params.gpu_index < 0 else f"cuda:{params.gpu_index}")

    model = torch.jit.load(params.ckpt_path, map_location=device).eval()

    pred_image_paths = glob(os.path.join(params.predict_dir, "*.nii.gz"))

    pred_dicts = tuple({"image": img} for img in pred_image_paths)

    keys = "image"
    pred_transforms = Compose(
        (
            LoadImaged(reader="NibabelReader", keys=keys),
            EnsureChannelFirstd(keys=keys),
            CustomResized(keys=keys, roi_size=roi_size),
            Orientationd(keys, axcodes="RAI"),
            NormalizeIntensityd(keys="image"),
            ToTensord(keys=keys),
        )
    )

    pred_ds = Dataset(pred_dicts, pred_transforms)

    saver = NiftiSaver(
        params.output_dir,
        output_dtype=np.uint8,
        dtype=np.float32,
        mode="nearest",
        padding_mode="zeros",
        separate_folder=False,
        print_log=True,  # make false for docker
    )

    sliding_inferer = SlidingWindowInferer(
        roi_size,
        params.sw_batch_size,
        params.sw_overlap,
        mode="gaussian",
        cache_roi_weight_map=True,
    )

    if params.post_process:
        from monai.transforms import KeepLargestConnectedComponent

        keep_connected_component = KeepLargestConnectedComponent(
            applied_labels=range(1, num_labels_with_bg),
            is_onehot=False,
            independent=True,
            connectivity=params.connectivity,
        )

    dl = DataLoader(
        pred_ds,
        batch_size=1,  # Because the images do not align and are not cropped
        num_workers=params.num_workers,
    )

    with torch.inference_mode():
        for batch in dl:
            image = batch["image"].to(device)

            output = sliding_inferer(image, model).cpu()

            # channel_dim = 1

            # sm = torch.softmax(output, dim=channel_dim)

            # print(
            #     "Mean Max with Back",
            #     sm.max(dim=channel_dim).values.mean(),
            # )

            # sm_fore = sm[:, 1:, ...].max(dim=channel_dim).values
            # print(
            #     "Mean Max without Back",
            #     sm_fore.mean(),
            # )

            # print(
            #     "Std Max without Back",
            #     sm_fore.std(),
            # )

            # print(
            #     "Max without Back",
            #     sm_fore.max(),
            # )

            # print(
            #     "Median without Back",
            #     sm_fore.median(),
            #     end="\n\n\n",
            # )

            # Squeezing for a single batch

            argmax_out = output.squeeze(0).argmax(0)

            post_out = (
                keep_connected_component(argmax_out)
                if params.post_process
                else argmax_out
            )

            meta_data = {k: v[0] for k, v in batch["image_meta_dict"].items()}
            saver(post_out, meta_data)

            # Run garbage collector if RAM is OOM
            # Reduced max GPU usage from 5G to 4G
            # gc.collect()
            # if params.gpu_index >= 0:
            #     torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--ckpt_path", default="flare_supervised_checkpoint.pt", type=str
    )
    parser.add_argument("--predict_dir", default="inputs", type=str)
    parser.add_argument("--output_dir", default="outputs", type=str)
    parser.add_argument("--sw_batch_size", default=8, type=int)
    parser.add_argument("--sw_overlap", default=0.25, type=float)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--gpu_index", default=0, type=int)
    parser.add_argument("--post_process", default=False, type=bool)
    parser.add_argument("--connectivity", default=None, type=int)

    args = parser.parse_args()

    main(args)
