from typing import Dict, Hashable, Mapping, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from monai.config import KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.transform import MapTransform, RandomizableTransform
from monai.utils.enums import TransformBackends


class SimulateLowResolution(RandomizableTransform):

    backend = [TransformBackends.TORCH]

    def __init__(
        self,
        zoom_range: Union[float, Tuple[float, float]] = (0.5, 1.0),
        prob: float = 0.1,
    ) -> None:
        RandomizableTransform.__init__(self, prob)
        self.zoom_range = (
            zoom_range if isinstance(zoom_range, tuple) else (zoom_range, 1)
        )

    def randomize(self, data):
        super().randomize(data)
        self._scale = self.R.uniform(*self.zoom_range)

    def __call__(self, img: NdarrayOrTensor, randomize: bool = True) -> NdarrayOrTensor:
        if randomize:
            self.randomize(None)
        if not self._do_transform:
            return img

        torch_img = torch.as_tensor(img)

        # self.orig_shape = np.asarray(img.shape)
        # self.output_shape = np.maximum(np.round(self._scale * self.orig_shape), 1)
        # downsampled: np.ndarray = resize(
        #     img,
        #     self.output_shape,
        #     order=self.order_downsample,
        #     mode="edge",
        #     anti_aliasing=False,
        # )
        downsampled = F.interpolate(
            torch_img.unsqueeze(0), scale_factor=self._scale, mode="nearest"
        )
        # Bicubic doesn't seem to work so, falling back to trilinear
        upsampled = F.interpolate(
            downsampled, size=torch_img.shape[1:], mode="trilinear"
        ).squeeze(0)
        # upsampled: np.ndarray = resize(
        #     downsampled,
        #     self.orig_shape,
        #     order=self.order_upsample,
        #     mode="edge",
        #     anti_aliasing=False,
        # )

        # Restore the initial type
        ret = upsampled.numpy() if isinstance(img, np.ndarray) else upsampled
        return ret


class SimulateLowResolutiond(RandomizableTransform, MapTransform):

    backend = SimulateLowResolution.backend

    def __init__(
        self,
        keys: KeysCollection,
        zoom_range: Union[float, Tuple[float, float]] = (0.5, 1.0),
        prob: float = 0.1,
        allow_missing_keys: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.backbone = SimulateLowResolution(zoom_range=zoom_range, prob=prob)

    def set_random_state(
        self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None
    ) -> "SimulateLowResolutiond":
        super().set_random_state(seed, state)
        self.backbone.set_random_state(seed, state)
        return self

    def __call__(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[Hashable, np.ndarray]:
        self.randomize(None)
        if not self._do_transform:
            return data

        d = dict(data)

        # Share same zoom range for all the keys
        self.backbone.randomize(None)
        for key in self.key_iterator(d):
            d[key] = self.backbone(d[key], randomize=False)
        return d
