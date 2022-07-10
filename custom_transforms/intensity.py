from typing import Dict, Hashable, Mapping, Optional, Tuple

import numpy as np
from monai.config import DtypeLike, KeysCollection
from monai.transforms import MapTransform, NormalizeIntensity

from . import NdarrayOrTensor


class NormalizeAndClipIntensityd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.NormalizeIntensity`.
    This transform can normalize only non-zero values or entire image,
    and can also calculate mean and std on each channel separately.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: monai.transforms.MapTransform
        subtrahend: the amount to subtract by (usually the mean)
        divisor: the amount to divide by (usually the standard deviation)
        nonzero: whether only normalize non-zero values.
        channel_wise: if True, calculate on each channel separately, otherwise,
            calculate on the entire image directly. default to False.
        dtype: output data type, if None, same as input image. defaults to float32.
        allow_missing_keys: don't raise exception if key is missing.
    """

    backend = NormalizeIntensity.backend

    def __init__(
        self,
        keys: KeysCollection,
        clip_range: Tuple[float, float] = (-2, 2),
        subtrahend: Optional[NdarrayOrTensor] = None,
        divisor: Optional[NdarrayOrTensor] = None,
        nonzero: bool = False,
        channel_wise: bool = False,
        dtype: DtypeLike = np.float32,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.normalizer = NormalizeIntensity(
            subtrahend, divisor, nonzero, channel_wise, dtype
        )
        self.clip_range = clip_range

    def __call__(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            norm = self.normalizer(d[key])
            d[key] = norm.clip(*self.clip_range)
        return d
