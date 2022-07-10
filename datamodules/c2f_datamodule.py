import math
from typing import Optional, Tuple

import numpy as np
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    HistogramNormalized,
    LoadImaged,
    Orientationd,
    RandRotated,
    RandSpatialCropSamplesd,
    RandZoomd,
    SpatialPadd,
    ToTensord,
)

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

    def get_transform(
        self,
        keys: TupleStr = BaseDataModule._dict_keys,
        do_random: bool = False,
    ):
        mode = "bilinear"
        additional_transforms = []
        zoom_mode = "trilinear"

        if not isinstance(keys, str):
            mode = (mode, "nearest")
            zoom_mode = (zoom_mode, "nearest")
            # additional_transforms.append(CastToTyped(keys="label", dtype=np.uint8))
            additional_transforms.append(CropForegroundd(keys=keys, source_key="label"))
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
                HistogramNormalized(keys="image", min=-1, max=1),
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
