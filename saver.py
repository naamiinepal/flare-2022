# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os.path
from typing import Optional, Union

import numpy as np
import torch
from monai.config import DtypeLike, PathLike
from monai.data import NibabelWriter
from monai.utils import GridSampleMode, GridSamplePadMode, InterpolateMode
from monai.utils import ImageMetaKey as Key


class NiftiSaver:
    """
    Save the image (in the form of torch tensor or numpy ndarray) and metadata dictionary into files.

    The name of saved file will be `{input_image_name}_{output_postfix}{output_ext}`,
    where the `input_image_name` is extracted from the provided metadata dictionary.
    If no metadata provided, a running index starting from 0 will be used as the filename prefix.

    Args:
        output_dir: output image directory.
        output_dtype: data type for saving data. Defaults to ``np.float32``.
        resample: whether to resample image (if needed) before saving the data array,
            based on the `spatial_shape` (and `original_affine`) from metadata.
        mode: This option is used when ``resample=True``. Defaults to ``"nearest"``.
            Depending on the writers, the possible options are
            - {``"bilinear"``, ``"nearest"``, ``"bicubic"``}.
              See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            - {``"nearest"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``, ``"area"``}.
              See also: https://pytorch.org/docs/stable/nn.functional.html#interpolate

        padding_mode: This option is used when ``resample = True``. Defaults to ``"border"``.
            Possible options are {``"zeros"``, ``"border"``, ``"reflection"``}
            See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
        dtype: data type during resampling computation. Defaults to ``np.float64`` for best precision.
            if None, use the data type of input data. To be compatible with other modules,
        squeeze_end_dims: if True, any trailing singleton dimensions will be removed (after the channel
            has been moved to the end). So if input is (C,H,W,D), this will be altered to (H,W,D,C), and
            then if C==1, it will be saved as (H,W,D). If D is also 1, it will be saved as (H,W). If `false`,
            image will always be saved as (H,W,D,C).
        data_root_dir: if not empty, it specifies the beginning parts of the input file's
            absolute path. It's used to compute `input_file_rel_path`, the relative path to the file from
            `data_root_dir` to preserve folder structure when saving in case there are files in different
            folders with the same file names. For example, with the following inputs:

            - input_file_name: `/foo/bar/test1/image.nii`
            - output_postfix: `seg`
            - output_ext: `.nii.gz`
            - output_dir: `/output`
            - data_root_dir: `/foo/bar`

            The output will be: /output/test1/image/image_seg.nii.gz

        separate_folder: whether to save every file in a separate folder. For example: for the input filename
            `image.nii`, postfix `seg` and folder_path `output`, if `separate_folder=True`, it will be saved as:
            `output/image/image_seg.nii`, if `False`, saving as `output/image_seg.nii`. Default to `True`.
        print_log: whether to print logs when saving. Default to `True`.
        channel_dim: the index of the channel dimension. Default to `0`.
            `None` to indicate no channel dimension.
    """

    def __init__(
        self,
        output_dir: PathLike = "./",
        output_postfix: str = "",
        output_dtype: DtypeLike = np.float32,
        resample: bool = True,
        mode: Union[GridSampleMode, InterpolateMode, str] = "nearest",
        padding_mode: Union[GridSamplePadMode, str] = GridSamplePadMode.BORDER,
        dtype: DtypeLike = np.float64,
        squeeze_end_dims: bool = True,
        data_root_dir: PathLike = "",
        separate_folder: bool = True,
        print_log: bool = True,
        channel_dim: Optional[int] = None,
    ):
        self.output_dir = output_dir
        self.output_postfix = output_postfix
        self.data_root_dir = data_root_dir
        self.separate_folder = separate_folder

        self.writer = NibabelWriter(output_dtype=output_dtype)

        self.data_kwargs = {
            "squeeze_end_dims": squeeze_end_dims,
            "channel_dim": channel_dim,
        }
        self.meta_kwargs = {
            "resample": resample,
            "mode": mode,
            "padding_mode": padding_mode,
            "dtype": dtype,
        }
        self.write_kwargs = {"verbose": print_log}
        self._data_index = 0

    def __call__(
        self, img: Union[torch.Tensor, np.ndarray], meta_data: Optional[dict] = None
    ):
        """
        Args:
            img: target data content that save into file. The image should be channel-first, shape: `[C,H,W,[D]]`.
            meta_data: key-value pairs of metadata corresponding to the data.
        """
        subject = meta_data[Key.FILENAME_OR_OBJ] if meta_data else str(self._data_index)
        patch_index = meta_data.get(Key.PATCH_INDEX) if meta_data else None

        output_path = self.path_extractor(subject, patch_index)

        self.writer.set_data_array(data_array=img, **self.data_kwargs)
        self.writer.set_metadata(meta_dict=meta_data, **self.meta_kwargs)
        self.writer.write(output_path, **self.write_kwargs)

        self._data_index += 1

    def path_extractor(self, filename: str, patch_index: Optional[int]) -> str:
        # get the filename and directory
        filedir, filename = os.path.split(filename)

        # remove extension
        filename, ext = filename.split(".", 1)

        # use data_root_dir to find relative path to file
        filedir_rel_path = ""
        if self.data_root_dir and filedir:
            filedir_rel_path = os.path.relpath(filedir, self.data_root_dir)

        # output folder path will be original name without the extension
        output = os.path.join(self.output_dir, filedir_rel_path)

        if self.separate_folder:
            output = os.path.join(output, filename)

        # create target folder if no existing
        os.makedirs(output, exist_ok=True)

        # ## Added Later On As Per Need ## #
        # Need to change test_0016_0000.nii.gz to test_0016.nii.gz
        filename = filename.rsplit("_", 1)[0]

        # Add folder and filename
        output = os.path.join(output, filename)

        # add the the postfix name
        if len(self.output_postfix) > 0:
            output += f"_{self.output_postfix}"

        if patch_index is not None:
            output += f"_{patch_index}"

        return os.path.normpath(f"{output}.{ext}")
