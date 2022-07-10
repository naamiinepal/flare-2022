import gc
import math
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
    HistogramNormalized,
    LoadImaged,
    Orientationd,
    ToTensord,
)
from torch.nn import functional as F

from custom_transforms import CustomResize, CustomResized
from saver import NiftiSaver

num_labels_with_bg = 14
intermediate_roi_size = (256, 256, 128)
coarse_roi_size = (128, 128, 64)
fine_roi_size = (192, 192, 96)
coarse_scale = np.asarray(intermediate_roi_size) / np.asarray(coarse_roi_size)
inverse_coarse_scale = tuple(1 / coarse_scale)

resize_fine = CustomResize(roi_size=fine_roi_size, image_only=True)


def main(params: Namespace):
    pred_image_paths = glob(os.path.join(params.predict_dir, "*.nii.gz"))

    pred_dicts = tuple({"image": img} for img in pred_image_paths)

    pred_transforms = Compose(
        (
            LoadImaged(reader="NibabelReader", keys="image"),
            EnsureChannelFirstd(keys="image"),
            CustomResized(keys="image", roi_size=intermediate_roi_size),
            Orientationd(keys="image", axcodes="RAI"),
            HistogramNormalized(keys="image", min=-1, max=1),
            ToTensord(keys="image"),
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
        channel_dim=0,
        print_log=params.verbose,  # make false for docker
    )

    device = torch.device("cpu" if params.gpu_index < 0 else f"cuda:{params.gpu_index}")

    coarse_sliding_inferer = SlidingWindowInferer(
        coarse_roi_size,
        params.sw_batch_size,
        params.sw_overlap,
        mode="gaussian",
        cache_roi_weight_map=True,
    )

    fine_sliding_inferer = SlidingWindowInferer(
        fine_roi_size,
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

        keep_connected_component_coarse = KeepLargestConnectedComponent(
            independent=False,
            connectivity=params.connectivity,
        )

    dl = DataLoader(
        pred_ds,
        batch_size=1,  # Because the images do not align and are not cropped
        num_workers=params.num_workers,
    )

    coarse_model = torch.jit.optimize_for_inference(
        torch.jit.load(params.coarse_ckpt_path, map_location=device)
    )
    fine_model = torch.jit.optimize_for_inference(
        torch.jit.load(params.fine_ckpt_path, map_location=device)
    )

    with torch.inference_mode():
        for batch in dl:
            image = batch["image"].to(device)
            coarse_image = F.interpolate(
                input=image,
                scale_factor=inverse_coarse_scale,
                mode="trilinear",
                recompute_scale_factor=True,
            )

            coarse_output: np.ndarray = (
                (coarse_sliding_inferer(coarse_image, coarse_model) >= 0.5)
                .squeeze(0)
                .cpu()
                .numpy()
            )

            final_output = torch.zeros(image.shape[1:], dtype=int, device=device)

            if coarse_output.any():
                if params.post_process:
                    coarse_output = keep_connected_component_coarse(coarse_output)
                x_indices, y_indices, z_indices = np.where(coarse_output.squeeze(0))

                # Multiplying less scale rather than all the indices
                x1 = int(x_indices.min() * coarse_scale[0])
                x2 = math.ceil(x_indices.max() * coarse_scale[0])
                y1 = int(y_indices.min() * coarse_scale[1])
                y2 = math.ceil(y_indices.max() * coarse_scale[1])
                z1 = int(z_indices.min() * coarse_scale[2])
                z2 = math.ceil(z_indices.max() * coarse_scale[2])
                cropped_image = image[0, :, x1 : x2 + 1, y1 : y2 + 1, z1 : z2 + 1]

                # Add batch to pass in the inferrer
                fine_image = resize_fine(cropped_image).unsqueeze(0)

                # (B, 14, H, W, D)
                # Convert to float for interpolation
                # Where to perfoc_outrm interpolation
                output = (
                    fine_sliding_inferer(fine_image, fine_model).argmax(
                        dim=1, keepdim=True
                    )
                    # .cpu()
                )
                scaled_out = F.interpolate(
                    input=output.float(),
                    size=cropped_image.shape[-3:],
                    mode="nearest",  # (-inf, inf) range of output
                ).squeeze(0)
                final_output[:, x1 : x2 + 1, y1 : y2 + 1, z1 : z2 + 1] = scaled_out

            argmax_out = final_output.cpu().numpy()

            if params.post_process:
                argmax_out = keep_connected_component(argmax_out)

            meta_data = {k: v[0] for k, v in batch["image_meta_dict"].items()}
            saver(argmax_out, meta_data)

            # Run garbage collector if RAM is OOM
            # Reduced max GPU usage from 5G to 4G
            gc.collect()
            if params.gpu_index >= 0:
                torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--coarse_ckpt_path", default="coarse_flare_model.ts", type=str)
    parser.add_argument("--fine_ckpt_path", default="fine_flare_model.ts", type=str)
    parser.add_argument("--predict_dir", default="inputs", type=str)
    parser.add_argument("--output_dir", default="outputs", type=str)
    parser.add_argument("--sw_batch_size", default=2, type=int)
    parser.add_argument("--sw_overlap", default=0.25, type=float)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--gpu_index", default=0, type=int)
    parser.add_argument("--post_process", default=True, type=bool)
    parser.add_argument("--connectivity", default=None, type=int)
    parser.add_argument("--verbose", default=False, type=bool)

    args = parser.parse_args()

    main(args)
