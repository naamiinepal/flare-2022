from typing import Dict, Hashable, Mapping, Tuple, Union

import numpy as np
from monai.config import KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.transform import MapTransform, RandomizableTransform
from monai.utils.enums import TransformBackends
from skimage.transform import resize


class SimulateLowResolutiond(RandomizableTransform, MapTransform):

    backend = [TransformBackends.NUMPY]

    def __init__(
        self,
        keys: KeysCollection,
        zoom_range: Union[float, Tuple[float, float]] = (0.5, 1.0),
        order_downsample: int = 0,
        order_upsample: int = 3,
        prob: float = 0.1,
        allow_missing_keys: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.zoom_range = (
            zoom_range if isinstance(zoom_range, tuple) else (zoom_range, 1)
        )

        self.order_downsample = order_downsample
        self.order_upsample = order_upsample

    def randomize(self):
        super().randomize(None)
        self._scale = self.R.uniform(*self.zoom_range)

    def __call__(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        self.randomize()
        if not self._do_transform:
            return d

        for key in self.key_iterator(d):
            img = d[key]
            self.orig_shape = np.asarray(img.shape)
            self.output_shape = np.maximum(np.round(self._scale * self.orig_shape), 1)
            downsampled: np.ndarray = resize(
                img,
                self.output_shape,
                order=self.order_downsample,
                mode="edge",
                anti_aliasing=False,
            )
            upsampled: np.ndarray = resize(
                downsampled,
                self.orig_shape,
                order=self.order_upsample,
                mode="edge",
                anti_aliasing=False,
            )
            d[key] = upsampled
        return d
