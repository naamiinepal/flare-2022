from copy import deepcopy
from typing import Dict, Hashable, Iterable, Mapping, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from monai.config import KeysCollection
from monai.transforms import InvertibleTransform, MapTransform, Transform
from monai.utils.enums import TraceKeys, TransformBackends

from . import NdarrayOrTensor


class CustomResize(Transform):
    """
    Resize the input image to given spatial size (with scaling, not cropping/padding).
    Implemented using :py:class:`torch.nn.functional.interpolate`.

    Args:
        spatial_size: expected shape of spatial dimensions after resize operation.
            if some components of the `spatial_size` are non-positive values,
            the transform will use the corresponding components of img size.
            For example, `spatial_size=(32, -1)` will be adapted
            to `(32, 64)` if the second spatial dimension size of img is `64`.
        size_mode: should be "all" or "longest", if "all", will use `spatial_size` for
            all the spatial dims, if "longest", rescale the image so that only the
            longest side is equal to specified `spatial_size`,
            which must be an int number in this case, keeping the aspect ratio of the
            initial image, refer to:
            https://albumentations.ai/docs/api_reference/augmentations/geometric/resize/
            #albumentations.augmentations.geometric.resize.LongestMaxSize.
        mode: {``"nearest"``, ``"nearest-exact"``, ``"linear"``, ``"bilinear"``,
            ``"bicubic"``, ``"trilinear"``, ``"area"``} The interpolation mode.
            Defaults to ``"area"``. See also:
            https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
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
                The interpolation mode. Defaults to ``self.mode``. See also:
                https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
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
        mode: {``"nearest"``, ``"nearest-exact"``, ``"linear"``, ``"bilinear"``,
            ``"bicubic"``, ``"trilinear"``, ``"area"``}
            The interpolation mode. Defaults to ``"area"``. See also:
            https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
            It also can be a sequence of string,
            each element corresponds to a key in ``keys``.
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
