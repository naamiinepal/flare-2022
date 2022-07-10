import os.path
from typing import Literal, Optional

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
    BinaryConvexHulld,
    BoundingMaskd,
    CustomResized,
)

from . import BaseDataModule, TupleStr


class CoarseDataModule(BaseDataModule):
    def __init__(
        self,
        transform: Optional[Literal["hull", "boundingmask"]] = None,
        supervised_dir=".",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        hull_str = transform if transform is not None else "binary"

        self.cache_dir = (
            f"{os.path.basename(supervised_dir)}_"
            f"{self.__class__.__name__}_{hull_str}_datacache"
        )

    def get_transform(
        self,
        keys: TupleStr = BaseDataModule._dict_keys,
        do_random: bool = False,
    ):
        mode = "bilinear"
        zoom_mode = "trilinear"
        additional_transforms = []
        if not isinstance(keys, str):
            mode = (mode, "nearest")
            zoom_mode = (zoom_mode, "nearest")
            # Compute hull directly if hull is needed, else make the image binary
            BinaryTransform = (
                BinaryConvexHulld if self.hparams.transform == "hull" else Binarized
            )
            additional_transforms.append(BinaryTransform(keys="label"))

        if do_random:
            additional_transforms.extend(self.get_weak_aug(keys, mode, zoom_mode))

        if not isinstance(keys, str) and self.hparams.transform == "boundingmask":
            additional_transforms.append(BoundingMaskd(keys="label"))

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
                # NormalizeAndClipIntensityd(keys="image"),
                HistogramNormalized(keys="image", min=-1, max=1),
                *additional_transforms,
                ToTensord(keys=keys),
            )
        )
