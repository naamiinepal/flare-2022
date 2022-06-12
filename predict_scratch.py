import gc
import os.path
from argparse import ArgumentParser
from glob import glob

import numpy as np
import torch
from monai.data import DataLoader, Dataset
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    # KeepLargestConnectedComponent,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    Spacingd,
    ToTensord,
)

# from saver import NiftiSaver

num_labels_with_bg = 14
roi_size = (128, 128, 64)


def main(params):
    device = torch.device("cpu" if params.gpu_index < 0 else f"cuda:{params.gpu_index}")

    model = torch.jit.load(params.ckpt_path, map_location=device).eval()

    pred_image_paths = glob(os.path.join(params.predict_dir, "*.nii.gz"))

    pred_dicts = tuple({"image": img} for img in pred_image_paths)

    pred_transforms = Compose(
        (
            LoadImaged(reader="NibabelReader", keys="image"),
            EnsureChannelFirstd(keys="image"),
            Spacingd(keys="image", pixdim=(2.5, 2.5, 2.5), dtype=np.float32),
            Orientationd(keys="image", axcodes="RAI"),
            NormalizeIntensityd(keys="image"),
            ToTensord(keys="image"),
        )
    )

    pred_ds = Dataset(pred_dicts, pred_transforms)

    # saver = NiftiSaver(
    #     params.output_dir,
    #     output_postfix="",
    #     mode="nearest",
    #     dtype=np.float32,
    #     output_dtype=np.uint8,
    #     separate_folder=False,
    #     print_log=True,  # make false for docker
    # )

    # keep_connected_component = KeepLargestConnectedComponent(
    #     applied_labels=range(1, num_labels_with_bg),
    #     is_onehot=False,
    #     independent=True,
    #     connectivity=1,
    # )

    dl = DataLoader(
        pred_ds,
        batch_size=1,  # Because the images do not align and are not cropped
        num_workers=params.num_workers,
    )

    with torch.inference_mode():
        for batch in dl:
            image = batch["image"].to(device)

            output = sliding_window_inference(
                image,
                roi_size,
                params.sw_batch_size,
                model,
                overlap=params.sw_overlap,
                # device="cpu",
                mode="gaussian",
            )

            channel_dim = 1

            print(
                "Mean Max",
                torch.softmax(output, dim=channel_dim)
                .max(dim=channel_dim)
                .values.mean(),
            )

            # Squeezing for a single batch

            # argmax_out = output.squeeze(0).argmax(dim=0)

            # connected_out = keep_connected_component(argmax_out)

            # meta_data = {k: v[0] for k, v in batch["image_meta_dict"].items()}
            # saver(connected_out, meta_data)

            # Run garbage collector if RAM is OOM
            # Reduced max GPU usage from 5G to 4G
            gc.collect()
            if params.gpu_index >= 0:
                torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--ckpt_path", default="flare_supervised_checkpoint.pt", type=str
    )
    parser.add_argument("--predict_dir", default="inputs", type=str)
    parser.add_argument("--output_dir", default="outputs", type=str)
    parser.add_argument("--sw_batch_size", default=16, type=int)
    parser.add_argument("--sw_overlap", default=0.25, type=float)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--gpu_index", default=0, type=int)
    args = parser.parse_args()

    main(args)
