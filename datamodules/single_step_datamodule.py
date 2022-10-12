from typing import Tuple

from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    ToTensord,
)

from custom_transforms import CustomResized
from . import BaseDataModule, TupleStr


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
        zoom_mode = "trilinear"
        if not isinstance(keys, str):
            mode = (mode, "nearest")
            zoom_mode = (zoom_mode, "nearest")

        additional_transforms = []
        if do_random:
            additional_transforms.extend(self.get_weak_aug(keys, mode, zoom_mode))

        return Compose(
            (
                LoadImaged(reader="NibabelReader", keys=keys),
                EnsureChannelFirstd(keys=keys),
                Orientationd(keys, axcodes="LPS"),
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
