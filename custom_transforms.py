from copy import deepcopy
from typing import Dict, Hashable, Iterable, Mapping, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from monai.config import KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms import (
    InvertibleTransform,
    MapTransform,
    RandomizableTransform,
    Transform,
)
from monai.utils.enums import TraceKeys, TransformBackends


class CustomResize(Transform):
    """
    Resize the input image to given spatial size (with scaling, not cropping/padding).
    Implemented using :py:class:`torch.nn.functional.interpolate`.

    Args:
        spatial_size: expected shape of spatial dimensions after resize operation.
            if some components of the `spatial_size` are non-positive values, the transform will use the
            corresponding components of img size. For example, `spatial_size=(32, -1)` will be adapted
            to `(32, 64)` if the second spatial dimension size of img is `64`.
        size_mode: should be "all" or "longest", if "all", will use `spatial_size` for all the spatial dims,
            if "longest", rescale the image so that only the longest side is equal to specified `spatial_size`,
            which must be an int number in this case, keeping the aspect ratio of the initial image, refer to:
            https://albumentations.ai/docs/api_reference/augmentations/geometric/resize/
            #albumentations.augmentations.geometric.resize.LongestMaxSize.
        mode: {``"nearest"``, ``"nearest-exact"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``, ``"area"``}
            The interpolation mode. Defaults to ``"area"``.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
    """

    backend = [TransformBackends.TORCH]

    def __init__(self, roi_size: Tuple[int, int, int], mode: str = "trilinear") -> None:
        self.roi_size = roi_size
        self.mode = mode

    def __call__(
        self, img: NdarrayOrTensor, mode: Optional[str] = None
    ) -> NdarrayOrTensor:
        """
        Args:
            img: channel first array, must have shape: (num_channels, H[, W, ..., ]).
            mode: {``"nearest"``, ``"nearest-exact"``, ``"linear"``,
                ``"bilinear"``, ``"bicubic"``, ``"trilinear"``, ``"area"``}
                The interpolation mode. Defaults to ``self.mode``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
        Raises:
            ValueError: When ``self.spatial_size`` length is less than ``img`` spatial dimensions.

        """
        img_t = torch.as_tensor(img)
        scale_factor = self.roi_size[0] / img_t.size(1)
        out_size = np.maximum(
            np.array(img_t.shape[1:]) * scale_factor,
            self.roi_size,
            casting="unsafe",
            dtype=int,
        )

        zoomed: torch.Tensor = F.interpolate(
            input=img_t.unsqueeze(0),
            size=tuple(out_size),
            mode=mode or self.mode,
        ).squeeze(0)

        # Retain original data type
        out = zoomed.numpy() if isinstance(img, np.ndarray) else zoomed
        return out


class CustomResized(MapTransform, InvertibleTransform):
    """
    Dictionary-based wrapper of :py:class:`CustomZoom`.

    Args:
        keys: Keys to pick data for transformation.
        mode: {``"nearest"``, ``"nearest-exact"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``, ``"area"``}
            The interpolation mode. Defaults to ``"area"``.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
            It also can be a sequence of string, each element corresponds to a key in ``keys``.
        allow_missing_keys: don't raise exception if key is missing.
        kwargs: other arguments for the `np.pad` or `torch.pad` function.
            note that `np.pad` treats channel dimension as the first dimension.

    """

    backend = CustomResize.backend

    def __init__(
        self,
        keys: KeysCollection,
        roi_size: Tuple[int, int, int],
        mode: Union[str, Iterable[str]] = "trilinear",
        allow_missing_keys: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(keys, allow_missing_keys)

        self.mode = (mode,) * len(self.keys) if isinstance(mode, str) else mode

        self.resizer = CustomResize(roi_size=roi_size, **kwargs)

    def __call__(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key, mode in self.key_iterator(d, self.mode):
            self.push_transform(
                d,
                key,
                extra_info={
                    "mode": mode,
                },
            )
            d[key] = self.resizer(
                d[key],
                mode=mode,
            )
        return d

    def inverse(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[Hashable, NdarrayOrTensor]:
        d = deepcopy(dict(data))
        for key in self.key_iterator(d):
            transform = self.get_most_recent_transform(d, key)
            # Create inverse transform
            orig_size = transform[TraceKeys.ORIG_SIZE]
            mode = transform[TraceKeys.EXTRA_INFO]["mode"]
            # Apply inverse
            d[key] = F.interpolate(
                input=d[key].unsqueeze(0),
                size=orig_size,
                mode=mode,
            ).squeeze(0)
            # Remove the applied transform
            self.pop_transform(d, key)
        return d


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
