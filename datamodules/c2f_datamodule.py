import math
import os
from typing import Optional, Tuple, List

import numpy as np
from glob import glob
from monai.data import Dataset
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    # HistogramNormalized,
    ScaleIntensityRanged,
    LoadImaged,
    Orientationd,
    RandRotated,
    RandSpatialCropSamplesd,
    RandZoomd,
    SpatialPadd,
    ToTensord,
)

from torch.utils.data import ConcatDataset
from custom_transforms import CustomResized


from . import BaseDataModule, TupleStr


class C2FDataModule(BaseDataModule):
    def __init__(
        self,
        num_labels_with_bg: int = 14,
        coarse_roi_size: Tuple[int, int, int] = (128, 128, 64),
        fine_roi_size: Tuple[int, int, int] = (192, 192, 96),
        intermediate_roi_size: Tuple[int, int, int] = (256, 256, 128),
        **kwargs
    ):
        super().__init__(roi_size=intermediate_roi_size, **kwargs)

        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None):
        if stage is None or stage == "fit" or stage == "validate":
            from sklearn.model_selection import train_test_split

            image_paths = self.get_supervised_image_paths("images")
            label_paths = self.get_supervised_image_paths("labels")

            data_dicts = tuple(
                {"image": img, "label": lab}
                for img, lab in zip(image_paths, label_paths)
            )

            train_files: List[dict]
            val_files: List[dict]

            train_files, val_files = train_test_split(
                data_dicts, test_size=self.hparams.val_ratio
            )

            if stage != "validate":
                train_transforms = self.get_transform(do_random=True)
                self.train_ds = self.get_dataset(train_files, train_transforms)

                if self.hparams.do_semi:
                    unlabeled_image_paths = glob(
                        os.path.join(self.hparams.semisupervised_dir, "*.nii.gz")
                    )
                    if isinstance(self.hparams.semi_mu, int):
                        # Sort by file size
                        unlabeled_image_paths.sort(key=lambda img: os.stat(img).st_size)

                        # Take only those images with smallest sizes
                        unlabeled_image_paths = unlabeled_image_paths[
                            : len(train_files) * self.hparams.semi_mu
                        ]

                    unlabeled_files = tuple(
                        {"image": img} for img in unlabeled_image_paths
                    )

                    unlabeled_transform = self.get_transform(
                        keys="image", do_random=True
                    )

                    unlabeled_ds = self.get_dataset(
                        unlabeled_files, unlabeled_transform
                    )

                    self.train_ds = ConcatDataset((self.train_ds, unlabeled_ds))

            val_transforms = self.get_transform(validate=True)

            self.val_ds = self.get_dataset(val_files, val_transforms)

        if stage is None or stage == "predict":
            from saver import NiftiSaver

            pred_image_paths = glob(os.path.join(self.hparams.predict_dir, "*.nii.gz"))
            pred_image_paths.sort()

            pred_dicts = tuple({"image": img} for img in pred_image_paths)

            pred_transforms = self.get_transform(keys="image")

            self.pred_ds = Dataset(pred_dicts, pred_transforms)

            self.saver = NiftiSaver(
                self.hparams.output_dir,
                output_postfix="",
                mode="nearest",
                dtype=np.float32,
                output_dtype=np.uint8,
                separate_folder=False,
                print_log=False,
                channel_dim=0,
            )

    def get_transform(
        self,
        keys: TupleStr = BaseDataModule._dict_keys,
        do_random: bool = False,
        validate: bool = False,
    ):
        mode = "bilinear"
        additional_transforms = []
        zoom_mode = "trilinear"

        if not isinstance(keys, str):
            mode = (mode, "nearest")
            zoom_mode = (zoom_mode, "nearest")
            # additional_transforms.append(CastToTyped(keys="label", dtype=np.uint8))
            if validate:
                roi_size = self.hparams.intermediate_roi_size
            else:
                additional_transforms.append(
                    CropForegroundd(keys=keys, source_key="label")
                )
                roi_size = self.hparams.fine_roi_size
        else:
            roi_size = self.hparams.intermediate_roi_size

        additional_transforms.append(
            CustomResized(keys=keys, roi_size=roi_size, mode=zoom_mode)
        )

        if do_random:
            additional_transforms.extend(self.get_weak_aug(keys, mode, zoom_mode))

        return Compose(
            (
                LoadImaged(reader="NibabelReader", keys=keys),
                EnsureChannelFirstd(keys=keys),
                Orientationd(keys=keys, axcodes="RAI"),
                ScaleIntensityRanged(
                    keys="image", a_min=-325, a_max=325, b_min=-1, b_max=1, clip=True
                ),  # To imitate the previous winner's preprocessing
                # HistogramNormalized(keys="image", min=-1, max=1),
                *additional_transforms,
                ToTensord(keys=keys),
            )
        )

    def get_weak_aug(
        self,
        keys: TupleStr,
        mode: TupleStr,
        zoom_mode: Optional[TupleStr] = None,
    ):
        roi_size = (
            self.hparams.fine_roi_size
            if not isinstance(keys, str)
            else self.hparams.intermediate_roi_size
        )

        if zoom_mode is None:
            zoom_mode = mode

        rot_angle = math.pi / 12

        return (
            RandRotated(
                keys=keys,
                range_x=rot_angle,
                range_y=rot_angle,
                range_z=rot_angle,
                dtype=np.float32,
                padding_mode="zeros",
                mode=mode,
                prob=0.9,
            ),
            RandZoomd(
                keys=keys,
                min_zoom=0.8,
                max_zoom=1.3,
                mode=zoom_mode,
                prob=0.9,
                padding_mode="constant",
                keep_size=False,  # Last spatial padd will handle the case
            ),
            RandSpatialCropSamplesd(
                keys=keys,
                roi_size=roi_size,
                num_samples=self.hparams.crop_num_samples,
                random_size=False,
            ),
            SpatialPadd(
                keys=keys,
                spatial_size=roi_size,
                mode="constant",
            ),
        )
