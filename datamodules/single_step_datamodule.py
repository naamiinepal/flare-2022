import math
from typing import Optional, Tuple

import numpy as np
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandRotated,
    RandSpatialCropSamplesd,
    RandZoomd,
    SpatialPadd,
    ToTensord,
)

from custom_transforms import CustomResized
from datamodules.basedatamodule import BaseDataModule, TupleStr


class SingleStepDataModule(BaseDataModule):
    def __init__(
        self,
        num_labels_with_bg: int,  # Needed by model
        roi_size: Tuple[int, int, int] = (128, 128, 64),
        **kwargs
    ):

        super().__init__(**kwargs)

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

        if do_random:
            additional_transforms.extend(self.get_weak_aug(keys, mode, zoom_mode))

        return Compose(
            (
                LoadImaged(reader="NibabelReader", keys=keys),
                EnsureChannelFirstd(keys=keys),
                Orientationd(keys, axcodes="RAI"),
                CustomResized(
                    keys=keys,
                    roi_size=self.hparams.roi_size,
                    mode=zoom_mode,
                ),
                NormalizeIntensityd(keys="image"),
                *additional_transforms,
                ToTensord(keys=keys),
            )
        )

    def get_weak_aug(
        self, keys: TupleStr, mode: TupleStr, zoom_mode: Optional[TupleStr] = None
    ):
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
                roi_size=self.hparams.roi_size,
                num_samples=self.hparams.crop_num_samples,
                random_size=False,
            ),
            SpatialPadd(keys=keys, spatial_size=self.hparams.roi_size),
        )
