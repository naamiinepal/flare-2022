import gc
import math

# import itk
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
from monai.data.box_utils import convert_box_mode
from torch.nn import functional as F

from custom_transforms import CustomResize, CustomResized
from saver import NiftiSaver

num_labels_with_bg = 14
intermediate_roi_size = (256, 256, 128)
coarse_roi_size = (128, 128, 64)
fine_roi_size = (192, 192, 96)
coarse_scale = np.asarray(intermediate_roi_size) / np.asarray(coarse_roi_size)
inverse_coarse_scale = tuple(1 / coarse_scale)
min_abdomen_size_mm = (125, 125, 320)  # in mm

resize_fine = CustomResize(roi_size=fine_roi_size, image_only=True)


# size_dir = "/mnt/HDD2/flare2022/abdomen_size_data"


# def save_nifti_from_array(img_array, filename):
#     filename = os.path.join(size_dir, f"{filename}.nii.gz")
#     itk_np_view = itk.image_view_from_array(img_array)
#     itk.imwrite(itk_np_view, filename)


def get_min_sized_abdomen(img_pix_dim, boxes_pix: torch.Tensor):
    """
    Returns the bounding box of the minimum sized abdomen in pixels.
    Extracts the physical size of the abdomen from the given bounding box.
    If the abdomen is too small, expands the bounding box to the minimum size,
    else does not change the bounding box.
    This function does not change the original resolution of the image.
    """
    # print(img_pix_dim, "Old box", boxes_pix)
    boxes_mm = torch.concat((boxes_pix[:3] * img_pix_dim, boxes_pix[3:] * img_pix_dim))

    xcenter, ycenter, zcenter, xsize, ysize, zsize = convert_box_mode(
        boxes_mm.unsqueeze(0), src_mode="xyzxyz", dst_mode="cccwhd"
    ).squeeze(0)
    xsize_min, ysize_min, zsize_min = np.maximum(
        min_abdomen_size_mm, (xsize, ysize, zsize)
    )
    xmin, ymin, zmin, xmax, ymax, zmax = (
        convert_box_mode(
            torch.tensor(
                [[xcenter, ycenter, zcenter, xsize_min, ysize_min, zsize_min]]
            ),
            src_mode="cccwhd",
            dst_mode="xyzxyz",
        )
        .squeeze(0)
        .numpy()
    )

    # convert back to pixel coordinates
    x1 = math.floor(max(0, boxes_pix[0] - abs(xmin)) / img_pix_dim[0])
    y1 = math.floor(max(0, boxes_pix[1] - abs(ymin)) / img_pix_dim[1])
    z1 = math.floor(max(0, boxes_pix[2] - abs(zmin)) / img_pix_dim[2])
    x2 = math.floor(xmax / img_pix_dim[0])
    y2 = math.floor(ymax / img_pix_dim[1])
    z2 = math.floor(zmax / img_pix_dim[2])

    # print("New box: ", x1, y1, z1, x2, y2, z2)
    return x1, y1, z1, x2, y2, z2


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
            # filename = (
            #     batch["image_meta_dict"]["filename_or_obj"][0]
            #     .split("/")[-1]
            #     .split(".")[0]
            # )
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
                # save_nifti_from_array(cropped_image, f"{filename}_cropped")
                # x1, y1, z1, x2, y2, z2 = get_min_sized_abdomen(
                #     img_pix_dim=batch["image_meta_dict"]["pixdim"][0][1:4].numpy(),
                #     boxes_pix=torch.tensor([x1, y1, z1, x2, y2, z2]),
                # )
                cropped_image = image[0, :, x1 : x2 + 1, y1 : y2 + 1, z1 : z2 + 1]
                # save_nifti_from_array(cropped_image, f"{filename}_resized")

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
    parser.add_argument(
        "--fine_ckpt_path", default="fine_flare_model_for_paper.ts", type=str
    )
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
