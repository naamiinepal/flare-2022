from copy import deepcopy
from typing import Dict, Hashable, Iterable, Mapping, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from monai.config import DtypeLike, KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms import (
    InvertibleTransform,
    MapTransform,
    NormalizeIntensity,
    RandomizableTransform,
    Transform,
)
from monai.utils.enums import TraceKeys, TransformBackends
from scipy.spatial import ConvexHull, Delaunay


class NormalizeAndClipIntensityd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.NormalizeIntensity`.
    This transform can normalize only non-zero values or entire image, and can also calculate
    mean and std on each channel separately.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: monai.transforms.MapTransform
        subtrahend: the amount to subtract by (usually the mean)
        divisor: the amount to divide by (usually the standard deviation)
        nonzero: whether only normalize non-zero values.
        channel_wise: if True, calculate on each channel separately, otherwise, calculate on
            the entire image directly. default to False.
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


class Binarized(MapTransform):

    backend = [TransformBackends.NUMPY, TransformBackends.TORCH]

    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)

    def __call__(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            lab = d[key]
            d[key] = lab.astype(bool) if isinstance(lab, np.ndarray) else lab.bool()
        return d


class BinaryConvexHull(MapTransform):

    backend = [TransformBackends.NUMPY]

    def __init__(
        self,
        keys: KeysCollection,
        retain_original: bool = False,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)

        self.retain_original = retain_original

    def __call__(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            # Remove the channel dimension
            img = d[key].squeeze(0)

            if isinstance(img, torch.Tensor):
                img = img.cpu().numpy()

            # Change to foreground and background only
            # Changing the data type creates a new copy
            # so no need to copy again
            bool_img = img.astype(bool)

            trans_pos_indices = np.vstack(np.where(bool_img)).T
            hull = ConvexHull(trans_pos_indices)

            triang = Delaunay(trans_pos_indices[hull.vertices])

            neg_indices = np.where(~bool_img)

            simp = triang.find_simplex(np.vstack(neg_indices).T) >= 0

            bool_img[neg_indices] = simp

            hull_key = f"{key}_hull" if self.retain_original else key

            d[hull_key] = np.expand_dims(bool_img, 0)
        return d


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

    def __init__(
        self,
        roi_size: Tuple[int, int, int],
        mode: str = "trilinear",
        image_only: bool = False,
    ) -> None:
        self.roi_size = roi_size
        self.mode = mode
        self.image_only = image_only

    def __call__(
        self,
        img: NdarrayOrTensor,
        original_affine: Optional[np.ndarray] = None,
        mode: Optional[str] = None,
    ) -> Union[NdarrayOrTensor, Tuple[NdarrayOrTensor, np.ndarray]]:
        """
        Args:
            img: channel first array, must have shape: (num_channels, H[, W, ..., ]).
            mode: {``"nearest"``, ``"nearest-exact"``, ``"linear"``,
                ``"bilinear"``, ``"bicubic"``, ``"trilinear"``, ``"area"``}
                The interpolation mode. Defaults to ``self.mode``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
        """
        img_t = torch.as_tensor(img)
        scale_factor = self.roi_size[0] / img_t.size(1)
        img_shape = np.array(img_t.shape[1:])
        out_size = np.maximum(
            img_shape * scale_factor,
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
        if self.image_only:
            return out

        affine_scaler = img_shape / out_size

        new_affine = (
            np.eye(img_t.ndim) if original_affine is None else np.copy(original_affine)
        )

        new_affine[: len(img_shape), : len(img_shape)] *= affine_scaler

        return out, new_affine


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
        meta_key_postfix: str = "meta_dict",
        **kwargs,
    ) -> None:
        super().__init__(keys, allow_missing_keys)

        self.mode = (mode,) * len(self.keys) if isinstance(mode, str) else mode

        self.resizer = CustomResize(roi_size=roi_size, **kwargs)
        self.meta_key_postfix = meta_key_postfix

    def __call__(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key, mode in self.key_iterator(d, self.mode):
            meta_dict = d[f"{key}_{self.meta_key_postfix}"]
            original_affine = meta_dict["affine"]

            d[key], meta_dict["affine"] = self.resizer(
                d[key],
                original_affine=original_affine,
                mode=mode,
            )
            self.push_transform(
                d,
                key,
                extra_info={
                    "mode": mode,
                    "original_affine": original_affine,
                },
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
            extra_info = transform[TraceKeys.EXTRA_INFO]
            mode = extra_info["mode"]
            # Apply inverse
            d[key] = F.interpolate(
                input=d[key].unsqueeze(0),
                size=orig_size,
                mode=mode,
            ).squeeze(0)

            # Restore original affine
            d[f"{key}_{self.meta_key_postfix}"]["spacing"] = extra_info[
                "original_affine"
            ]

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
