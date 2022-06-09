import gc
import os.path
from argparse import ArgumentParser
from glob import glob

# import numpy as np
import torch
from monai.data import DataLoader, Dataset
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    Spacingd,
    ToTensord,
)

# from saver import NiftiSaver


def main(params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.jit.load("flare_supervised_checkpoint.pt", map_location=device).eval()

    pred_image_paths = glob(os.path.join(params.predict_dir, "*.nii.gz"))[3:]

    pred_dicts = tuple({"image": img} for img in pred_image_paths)

    pred_transforms = Compose(
        (
            LoadImaged(reader="NibabelReader", keys="image"),
            EnsureChannelFirstd(keys="image"),
            Spacingd(keys="image", pixdim=(2.5, 2.5, 2.5)),
            Orientationd(keys="image", axcodes="RAI"),
            NormalizeIntensityd(keys="image"),
            ToTensord(keys="image"),
        )
    )

    pred_ds = Dataset(pred_dicts, pred_transforms)

    # saver = NiftiSaver(
    #     params.output_dir,
    #     output_postfix="",
    #     output_dtype=np.uint8,
    #     separate_folder=False,
    #     print_log=True,  # make false for docker
    # )

    roi_size = (128, 128, 64)

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
            )

            channel_dim = 1

            print(
                "Max",
                torch.softmax(output, dim=channel_dim)
                .max(dim=channel_dim)
                .values.mean(),
            )

            # Squeezing for a single batch
            # argmax_out: np.ndarray = output.squeeze(0).argmax(dim=0).numpy()
            # meta_data = {k: v[0] for k, v in batch["image_meta_dict"].items()}
            # saver(argmax_out, meta_data)

            # Run garbage collector if RAM is OOM
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--predict_dir", default="inputs", type=str)
    parser.add_argument("--output_dir", default="outputs", type=str)
    parser.add_argument("--sw_batch_size", default=16, type=int)
    parser.add_argument("--sw_overlap", default=0.25, type=float)
    parser.add_argument("--num_workers", default=4, type=int)
    args = parser.parse_args()

    main(args)
