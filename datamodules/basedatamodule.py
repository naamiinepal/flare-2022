import math
import os
from glob import glob
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
import pytorch_lightning as pl
from monai.data import CacheDataset, DataLoader, Dataset, PersistentDataset
from monai.transforms import (
    Compose,
    RandAdjustContrast,
    RandGaussianNoise,
    RandGaussianSmooth,
    RandScaleIntensity,
)
from torch.utils.data import ConcatDataset

from custom_transforms import SimulateLowResolution

TupleStr = Union[Tuple[str, str], str]


class BaseDataModule(pl.LightningDataModule):

    _dict_keys = ("image", "label")

    def __init__(
        self,
        supervised_dir: str = ".",
        semisupervised_dir: str = ".",
        predict_dir: str = ".",
        output_dir: str = ".",
        val_ratio: float = 0.2,
        do_semi: bool = False,
        semi_mu: Optional[int] = None,
        crop_num_samples: int = 4,
        batch_size: int = 16,
        ds_cache_type: Optional[Literal["mem", "disk"]] = None,
        max_workers: int = 4,
        pin_memory: bool = True,
        roi_size: Optional[Tuple[int, int, int]] = [128, 128, 64],
        **kwargs,
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
                train_transforms = self.get_transform(do_random=True)
                self.train_ds = self.get_dataset(train_files, train_transforms)

                if self.hparams.do_semi:
                    unlabeled_image_paths = glob(
                        os.path.join(self.hparams.semisupervised_dir, "*.nii.gz")
                    )
                    if isinstance(self.hparams.semi_mu, int):
                        # Sort by file size
                        unlabeled_image_paths.sort(key=lambda img: os.stat(img).st_size)

                        # Take only those images with smallest sizes
                        unlabeled_image_paths = unlabeled_image_paths[
                            : len(train_files) * self.hparams.semi_mu
                        ]

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
            from saver import NiftiSaver

            pred_image_paths = glob(os.path.join(self.hparams.predict_dir, "*.nii.gz"))
            pred_image_paths.sort()

            pred_dicts = tuple({"image": img} for img in pred_image_paths)

            pred_transforms = self.get_transform(keys="image")

            self.pred_ds = Dataset(pred_dicts, pred_transforms)

            self.saver = NiftiSaver(
                self.hparams.output_dir,
                output_postfix="",
                mode="nearest",
                dtype=np.float32,
                output_dtype=np.uint8,
                separate_folder=False,
                print_log=False,
            )
        # NOTE: This is a temp workaround to view validation results, delete later
        from saver import NiftiSaver
        self.saver = NiftiSaver(
                self.hparams.output_dir,
                output_postfix="",
                mode="nearest",
                dtype=np.float32,
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

    @staticmethod
    def get_strong_aug():
        return Compose(
            (
                RandGaussianNoise(prob=0.9),
                RandGaussianSmooth(
                    # sigma_x=(0.5, 1.5),
                    # sigma_y=(0.5, 1.5),
                    # sigma_z=(0.5, 1.5),
                    prob=0.9,
                ),
                RandScaleIntensity(factors=(0.7, 1.3), prob=0.9),
                RandAdjustContrast(gamma=(0.65, 1.5), prob=0.9),
                SimulateLowResolution(zoom_range=0.5, prob=0.9),
            )
        )

    def get_dataset(self, *dataset_args):
        if self.hparams.ds_cache_type == "mem":
            return CacheDataset(*dataset_args, num_workers=self.num_workers)
        elif self.hparams.ds_cache_type == "disk":
            return PersistentDataset(
                *dataset_args,
                cache_dir=(
                    f"{os.path.basename(self.hparams.supervised_dir)}_"
                    f"{self.__class__.__name__}_datacache"
                ),
                pickle_protocol=5,
            )
        return Dataset(*dataset_args)
