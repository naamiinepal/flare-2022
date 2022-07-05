from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    HistogramNormalized,
    LoadImaged,
    Orientationd,
    ToTensord,
)

from custom_transforms import (  # NormalizeAndClipIntensityd,
    Binarized,
    BinaryConvexHull,
    CustomResized,
    NormalizeAndClipIntensityd,
)
from datamodules.basedatamodule import BaseDataModule, TupleStr


class CoarseDataModule(BaseDataModule):
    def __init__(self, use_hull: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters()

    def get_transform(
        self,
        keys: TupleStr = BaseDataModule._dict_keys,
        do_random: bool = False,
    ):
        mode = "bilinear"
        zoom_mode = "trilinear"
        additional_transforms = tuple()
        if not isinstance(keys, str):
            mode = (mode, "nearest")
            zoom_mode = (zoom_mode, "nearest")
            BinaryTransform = BinaryConvexHull if self.hparams.use_hull else Binarized
            additional_transforms = (BinaryTransform(keys="label"),)

        if do_random:
            additional_transforms = (
                *additional_transforms,
                *self.get_weak_aug(keys, mode, zoom_mode),
            )

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
                NormalizeAndClipIntensityd(keys="image"),
                *additional_transforms,
                ToTensord(keys=keys),
            )
        )
