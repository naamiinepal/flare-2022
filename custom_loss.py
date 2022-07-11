from typing import Iterable, List, Tuple, TypeVar

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt as eucl_distance
from torch import Tensor, einsum
from torch.nn import Module

NdArrayorTensor = TypeVar("NdArrayorTensor", np.ndarray, Tensor)


class SurfaceLoss(Module):
    def __init__(self, idc: List[int]):
        # Self.idc is used to filter out some classes of the target mask.
        # Use fancy indexing
        self.idc = idc

    def __call__(self, probs: Tensor, dist_maps: Tensor) -> Tensor:
        assert self.simplex(probs)
        assert not self.one_hot(dist_maps)

        pc = probs[:, self.idc, ...]
        dc = dist_maps[:, self.idc, ...]

        loss = einsum("bkxyz,bkxyz->bkxyz", pc, dc).mean()

        return loss

    @staticmethod
    def class2one_hot(seg: Tensor, K: int) -> Tensor:
        # Breaking change but otherwise can't deal with both 2d and 3d
        # if len(seg.shape) == 3:  # Only w, h, d, used by the dataloader
        #     return class2one_hot(seg.unsqueeze(dim=0), K)[0]

        assert SurfaceLoss.sset(seg, range(K)), (SurfaceLoss.uniq(seg), K)

        b, *img_shape = seg.shape

        res = torch.zeros((b, K, *img_shape), device=seg.device).scatter_(
            1, seg.unsqueeze(1), 1
        )

        assert res.shape == (b, K, *img_shape)
        assert SurfaceLoss.one_hot(res)

        return res

    @staticmethod
    def gt_transform(img: NdArrayorTensor, K: int):
        # Add one dimension to simulate batch
        seg = torch.as_tensor(img).unsqueeze(0)
        # Then pop the element to go back to img shape
        res = SurfaceLoss.class2one_hot(seg, K).squeeze(0)
        return res

    @staticmethod
    def dist_map_transform(resolution: Tuple[float, ...], K: int):
        def func(img: NdArrayorTensor):
            seg = SurfaceLoss.gt_transform(img, K)
            nd = SurfaceLoss.one_hot2dist(seg, resolution)
            return nd

        return func

    @staticmethod
    def simplex(t: NdArrayorTensor, axis=1):
        _sum = t.sum(axis)

        numel = t.numel() if isinstance(t, torch.Tensor) else t.size

        return int(_sum.sum() == numel)

    @staticmethod
    def uniq(a: NdArrayorTensor):
        uniques = (
            torch.unique(a).cpu().numpy()
            if isinstance(a, torch.Tensor)
            else np.unique(a)
        )

        return set(uniques)

    @staticmethod
    def sset(a: NdArrayorTensor, sub: Iterable) -> bool:
        return SurfaceLoss.uniq(a).issubset(sub)

    @staticmethod
    def one_hot(t: NdArrayorTensor, axis=1) -> bool:
        return SurfaceLoss.simplex(t, axis) and SurfaceLoss.sset(t, (0, 1))

    @staticmethod
    def one_hot2dist(
        seg: NdArrayorTensor,
        resolution: Tuple[float, float, float] = None,
        retain_device: bool = False,
    ) -> NdArrayorTensor:
        assert SurfaceLoss.one_hot(seg, axis=0)

        _seg = seg if isinstance(seg, np.ndarray) else seg.cpu().numpy()

        _seg = _seg.astype(np.bool)

        res = np.zeros_like(_seg)
        for k, posmask in enumerate(_seg):
            if posmask.any():
                negmask = ~posmask
                res[k] = (
                    eucl_distance(negmask, sampling=resolution) * negmask
                    - (eucl_distance(posmask, sampling=resolution) - 1) * posmask
                )
            # The idea is to leave blank the negative classes
            # since this is one-hot encoded, another class will supervise that pixel

        return (
            res
            if isinstance(seg, np.ndarray)
            else torch.as_tensor(res, device=seg.device if retain_device else None)
        )
