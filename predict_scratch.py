import os.path
from argparse import ArgumentParser
from glob import glob

import torch
from monai.data import DataLoader, Dataset
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    NormalizeIntensityd,
    ToTensord,
)

from saver import NiftiSaver


def main(params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.jit.load("abdomen_checkpoint.pt", map_location=device).eval()

    pred_image_paths = glob(os.path.join(params.predict_dir, "*.nii.gz"))

    pred_dicts = tuple({"image": img} for img in pred_image_paths)

    pred_transforms = Compose(
        (
            LoadImaged(keys="image"),
            EnsureChannelFirstd(keys="image"),
            NormalizeIntensityd(keys="image"),
            ToTensord(keys="image"),
        )
    )

    pred_ds = Dataset(pred_dicts, pred_transforms)

    saver = NiftiSaver(
        params.output_dir,
        output_postfix="",
        separate_folder=False,
        print_log=False,
    )

    roi_size = (128, 128, 32)

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
                device="cpu",
            )

            batch_meta_data = batch["image_meta_dict"]

            for i, out in enumerate(output):
                argmax_out = out.argmax(dim=0)
                meta_data = {k: v[i] for k, v in batch_meta_data.items()}
                saver(argmax_out, meta_data)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--predict_dir", default="inputs", type=str)
    parser.add_argument("--output_dir", default="outputs", type=str)
    parser.add_argument("--sw_batch_size", default=16, type=int)
    parser.add_argument("--sw_overlap", default=0.25, type=float)
    parser.add_argument("--num_workers", default=4, type=int)
    args = parser.parse_args()

    main(args)
