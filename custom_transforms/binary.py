from typing import Dict, Hashable, Mapping

import numpy as np
import torch
from monai.config import KeysCollection
from monai.transforms import MapTransform
from monai.utils.enums import TransformBackends
from scipy.spatial import ConvexHull, Delaunay

from . import NdarrayOrTensor


class Binarized(MapTransform):

    backend = [TransformBackends.NUMPY, TransformBackends.TORCH]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = d[key] > 0
        return d


class BinaryConvexHulld(MapTransform):

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
            orig_img = d[key]

            img = orig_img.squeeze(0)

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

            hull_img = np.expand_dims(bool_img, 0).astype(np.float32)

            if isinstance(orig_img, torch.Tensor):
                hull_img = torch.as_tensor(hull_img, device=orig_img.device)

            hull_key = f"{key}_hull" if self.retain_original else key
            d[hull_key] = hull_img
        return d


class BoundingMaskd(MapTransform):
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
            orig_label = d[key]

            backbone = torch if isinstance(orig_label, torch.Tensor) else np

            *_, x_indices, y_indices, z_indices = backbone.where(orig_label)

            # Multiplying less scale rather than all the indices
            x1 = x_indices.min()
            x2 = x_indices.max()

            y1 = y_indices.min()
            y2 = y_indices.max()

            z1 = z_indices.min()
            z2 = z_indices.max()

            mask_img = backbone.zeros_like(orig_label, dtype=bool)

            mask_img[..., x1 : x2 + 1, y1 : y2 + 1, z1 : z2 + 1] = True

            d[key] = mask_img
        return d
