import gc
import math
import os.path
from argparse import ArgumentParser, Namespace
from glob import glob
import sys

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

from custom_transforms import CustomResized
from saver import NiftiSaver

# from saver import NiftiSaver

num_labels_with_bg = 14
intermediate_roi_size = (256, 256, 128)
coarse_roi_size = (128, 128, 64)
fine_roi_size = (192, 192, 96)
coarse_scale = np.asarray(intermediate_roi_size) / np.asarray(coarse_roi_size)


def resize_fine(image: torch.Tensor) -> torch.Tensor:
    return F.interpolate(
        input=image.unsqueeze(0),
        size=fine_roi_size,
        mode="trilinear",
    ).squeeze(0)


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
        print_log=params.verbose,  # make false for docker
    )

    device = torch.device("cpu" if params.gpu_index < 0 else f"cuda:{params.gpu_index}")

    sliding_inferer = SlidingWindowInferer(
        intermediate_roi_size,
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
            is_onehot=True,
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

    def forward(image: torch.Tensor):
        coarse_image = F.interpolate(
            input=image,
            size=coarse_roi_size,
            mode="trilinear",
        )

        coarse_output: np.ndarray = (coarse_model(coarse_image) >= 0.5).cpu().numpy()

        img_shape = image.shape
        final_output = (
            torch.tensor(
                (sys.maxsize, *((-sys.maxsize,) * (num_labels_with_bg - 1))),
                dtype=torch.float32,
                device=device,
            )
            .expand((img_shape[0], *img_shape[2:], num_labels_with_bg))
            .movedim(-1, 1)
            .clone()  # To have different memory locations
        )

        has_mask = coarse_output.any()
        if has_mask:
            cropped_images = []
            cropped_indices = []

            for c_out, img in zip(coarse_output, image):
                if params.post_process:
                    c_out = keep_connected_component_coarse(c_out)
                c_out = c_out.squeeze(0)
                x_indices, y_indices, z_indices = np.where(c_out)

                # Multiplying less scale rather than all the indices
                x1 = int(x_indices.min() * coarse_scale[0])
                x2 = math.ceil(x_indices.max() * coarse_scale[0])
                y1 = int(y_indices.min() * coarse_scale[1])
                y2 = math.ceil(y_indices.max() * coarse_scale[1])
                z1 = int(z_indices.min() * coarse_scale[2])
                z2 = math.ceil(z_indices.max() * coarse_scale[2])
                cropped_indices.append((x1, x2, y1, y2, z1, z2))
                cropped_images.append(img[:, x1 : x2 + 1, y1 : y2 + 1, z1 : z2 + 1])

            # (B, 1, H, W, D)
            fine_image = torch.stack(tuple(map(resize_fine, cropped_images)))

            # (B, 14, H, W, D)
            # Convert to float for interpolation
            # Where to perfoc_outrm interpolation
            output = fine_model(fine_image).cpu()
            for i, (o, crop_img, ind) in enumerate(
                zip(output, cropped_images, cropped_indices)
            ):
                scaled_out = F.interpolate(
                    input=o.unsqueeze(0),
                    size=crop_img.shape[1:],
                    mode="trilinear",  # (-inf, inf) range of output
                ).squeeze(0)
                x1, x2, y1, y2, z1, z2 = ind
                final_output[i][:, x1 : x2 + 1, y1 : y2 + 1, z1 : z2 + 1] = scaled_out
        return final_output

    with torch.inference_mode():
        for batch in dl:
            image = batch["image"].to(device)

            output = sliding_inferer(image, forward)

            argmax_out = output.squeeze(0).argmax(0).cpu().numpy()

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
