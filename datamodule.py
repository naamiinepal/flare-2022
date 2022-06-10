import math
import os.path
import random
from glob import glob
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
import pytorch_lightning as pl
from monai.data import CacheDataset, DataLoader, Dataset, PersistentDataset
from monai.transforms import (
    CastToTyped,
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    Rand3DElasticd,
    RandAdjustContrastd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    RandSpatialCropSamplesd,
    Spacingd,
    SpatialPadd,
    ToTensord,
)
from torch.utils.data import ConcatDataset

from custom_transforms import SimulateLowResolutiond

TupleStr = Union[Tuple[str, str], str]


class DataModule(pl.LightningDataModule):

    _dict_keys = ("image", "label")

    def __init__(
        self,
        num_labels_with_bg: Optional[int] = None,
        supervised_dir: str = ".",
        semisupervised_dir: str = ".",
        predict_dir: str = ".",
        output_dir: str = ".",
        val_ratio: float = 0.2,
        do_semi: float = False,
        semi_mu: Optional[int] = None,
        crop_num_samples: int = 4,
        batch_size: int = 16,
        ds_cache_type: Optional[Literal["mem", "disk"]] = None,
        max_workers: int = 4,
        roi_size: Tuple[int, int, int] = (128, 128, 64),
        pixdim: Tuple[float, float, float] = (3, 3, 1),
        pin_memory: bool = True,
        **kwargs
    ):
        super().__init__()

        self.save_hyperparameters()

        self.num_workers = min(os.cpu_count(), max_workers)

    def setup(self, stage: Optional[str] = None):
        if stage is None or stage == "fit" or stage == "validate":
            from sklearn.model_selection import train_test_split

            image_paths = self.get_supervised_image_paths("images")
            label_paths = self.get_supervised_image_paths("labels")

            data_dicts = tuple(
                {"image": img, "label": lab}
                for img, lab in zip(image_paths, label_paths)
            )

            train_files: List[dict]
            val_files: List[dict]

            train_files, val_files = train_test_split(
                data_dicts, test_size=self.hparams.val_ratio
            )

            if stage != "validate":
                assert (
                    self.hparams.num_labels_with_bg is not None
                ), "Number of Labels is needed for training"

                train_transforms = self.get_transform(do_random=True)
                self.train_ds = self.get_dataset(train_files, train_transforms)

                if self.hparams.do_semi:
                    unlabeled_image_paths = glob(
                        os.path.join(self.hparams.semisupervised_dir, "*.nii.gz")
                    )
                    if isinstance(self.hparams.semi_mu, int):
                        unlabeled_image_paths = random.sample(
                            unlabeled_image_paths,
                            k=len(train_files) * self.hparams.semi_mu,
                        )

                    unlabeled_files = tuple(
                        {"image": img} for img in unlabeled_image_paths
                    )

                    unlabeled_transform = self.get_transform(
                        keys="image", do_random=True
                    )

                    unlabeled_ds = self.get_dataset(
                        unlabeled_files, unlabeled_transform
                    )

                    self.train_ds = ConcatDataset((self.train_ds, unlabeled_ds))

            val_transforms = self.get_transform()

            self.val_ds = self.get_dataset(val_files, val_transforms)

        if stage is None or stage == "predict":
            import numpy as np

            from saver import NiftiSaver

            pred_image_paths = glob(os.path.join(self.hparams.predict_dir, "*.nii.gz"))
            pred_image_paths.sort()

            pred_dicts = tuple({"image": img} for img in pred_image_paths)

            pred_transforms = self.get_transform(keys="image")

            self.pred_ds = Dataset(pred_dicts, pred_transforms)

            self.saver = NiftiSaver(
                self.hparams.output_dir,
                output_postfix="",
                output_dtype=np.uint8,
                separate_folder=False,
                print_log=False,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=math.ceil(
                self.hparams.batch_size / self.hparams.crop_num_samples
            ),
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.hparams.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=1,  # Because the images do not align and are not cropped
            num_workers=self.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.pred_ds,
            batch_size=1,  # Because the images do not align and are not cropped
            num_workers=self.num_workers,
        )

    def get_supervised_image_paths(self, baseDir: str):
        image_paths = glob(
            os.path.join(self.hparams.supervised_dir, baseDir, "*.nii.gz")
        )
        image_paths.sort()
        return image_paths

    def get_transform(
        self,
        keys: TupleStr = _dict_keys,
        do_random: bool = False,
    ):
        mode = "bilinear"
        additional_transforms = []

        if len(keys) == 2:
            mode = (mode, "nearest")
            additional_transforms.append(CastToTyped(keys="label", dtype=np.uint8))

        if do_random:
            additional_transforms.extend(self.get_weak_aug(keys, mode))

        return Compose(
            (
                LoadImaged(reader="NibabelReader", keys=keys),
                EnsureChannelFirstd(keys=keys),
                Spacingd(
                    keys=keys, pixdim=self.hparams.pixdim, mode=mode, dtype=np.float32
                ),
                Orientationd(keys, axcodes="RAI"),
                NormalizeIntensityd(keys="image"),
                *additional_transforms,
                ToTensord(keys=keys),
            )
        )

    def get_weak_aug(self, keys: TupleStr, mode: TupleStr):
        return (
            RandGaussianNoised(
                keys="image",
                prob=0.15,
            ),
            RandGaussianSmoothd(
                keys="image",
                sigma_x=(0.5, 1.5),
                sigma_y=(0.5, 1.5),
                sigma_z=(0.5, 1.5),
                prob=0.2,
            ),
            RandScaleIntensityd(keys="image", factors=(0.7, 1.3), prob=0.15),
            RandAdjustContrastd(keys="image", gamma=(0.65, 1.5), prob=0.15),
            RandSpatialCropSamplesd(
                keys=keys,
                roi_size=self.hparams.roi_size,
                num_samples=self.hparams.crop_num_samples,
                random_size=False,
            ),
            SpatialPadd(keys=keys, spatial_size=self.hparams.roi_size),
        )

    @staticmethod
    def get_strong_aug():
        return Compose(
            (
                Rand3DElasticd(
                    keys=DataModule._dict_keys,
                    sigma_range=(9, 13),
                    magnitude_range=(0, 900),
                    rotate_range=(math.pi / 12, math.pi / 12, math.pi / 12),
                    scale_range=((0.85, 1.25), (0.85, 1.25), (0.85, 1.25)),
                    padding_mode="zeros",
                    mode=("bilinear", "nearest"),
                    prob=0.8,
                ),
                SimulateLowResolutiond(keys="image", zoom_range=0.5, prob=0.25),
            )
        )

    def get_dataset(self, *dataset_args):
        if self.hparams.ds_cache_type == "mem":
            return CacheDataset(*dataset_args, num_workers=self.num_workers)
        elif self.hparams.ds_cache_type == "disk":
            return PersistentDataset(
                *dataset_args,
                cache_dir=os.path.basename(self.hparams.supervised_dir) + "_datacache",
                pickle_protocol=5
            )
        return Dataset(*dataset_args)
