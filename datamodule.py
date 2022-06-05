import math
import os.path
from glob import glob
from typing import Literal, Optional, Tuple

import pytorch_lightning as pl
from monai.data import CacheDataset, DataLoader, Dataset, PersistentDataset
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    LoadImaged,
    NormalizeIntensityd,
    Rand3DElasticd,
    RandAdjustContrastd,
    RandCropByLabelClassesd,
    RandGaussianNoised,
    RandGaussianSharpend,
    RandScaleIntensityd,
    Spacingd,
    ToTensord,
)


class DataModule(pl.LightningDataModule):

    _dict_keys = ("image", "label")

    def __init__(
        self,
        num_labels_with_bg: Optional[int] = None,
        supervised_dir: str = ".",
        predict_dir: str = ".",
        output_dir: str = ".",
        val_ratio: float = 0.2,
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

            images = self.get_supervised_image_paths("images")
            labels = self.get_supervised_image_paths("labels")

            data_dicts = tuple(
                {"image": img, "label": lab} for img, lab in zip(images, labels)
            )

            train_files, val_files = train_test_split(
                data_dicts, test_size=self.hparams.val_ratio
            )

            if stage != "validate":
                assert (
                    self.hparams.num_labels_with_bg is not None
                ), "Number of Labels is needed for training"

                keys = self._dict_keys
                train_transforms = self.get_transform(
                    Rand3DElasticd(
                        keys=keys,
                        sigma_range=(5, 7),
                        magnitude_range=(50, 150),
                        padding_mode="zeros",
                        rotate_range=(math.pi / 6, math.pi / 6, math.pi / 6),
                        scale_range=((0.7, 1.4), (0.7, 1.4), (0.7, 1.4)),
                        prob=0.35,
                        mode=("bilinear", "nearest"),
                    ),
                    RandGaussianNoised(
                        keys="image",
                        prob=0.15,
                    ),
                    RandScaleIntensityd(keys="image", factors=(0.65, 1.5), prob=0.15),
                    RandAdjustContrastd(keys="image", gamma=(0.7, 1.5), prob=0.15),
                    RandGaussianSharpend(
                        keys="image",
                        sigma1_x=(0.5, 1.5),
                        sigma1_y=(0.5, 1.5),
                        sigma1_z=(0.5, 1.5),
                        prob=0.2,
                    ),
                    RandCropByLabelClassesd(
                        keys=keys,
                        label_key="label",
                        spatial_size=self.hparams.roi_size,
                        num_samples=self.hparams.crop_num_samples,
                        num_classes=self.hparams.num_labels_with_bg,
                    ),
                )
                self.train_ds = self.get_dataset(train_files, train_transforms)

            val_transforms = self.get_transform()

            self.val_ds = self.get_dataset(val_files, val_transforms)

        if stage is None or stage == "predict":
            from saver import NiftiSaver

            pred_image_paths = glob(os.path.join(self.hparams.predict_dir, "*.nii.gz"))
            pred_image_paths.sort()

            pred_dicts = tuple({"image": img} for img in pred_image_paths)

            keys = "image"
            pred_transforms = Compose(
                (
                    LoadImaged(keys=keys),
                    EnsureChannelFirstd(keys=keys),
                    NormalizeIntensityd(keys="image"),
                    ToTensord(keys=keys),
                ),
            )

            self.pred_ds = Dataset(pred_dicts, pred_transforms)

            self.saver = NiftiSaver(
                self.hparams.output_dir,
                output_postfix="",
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

    def get_transform(self, *random_transforms):
        keys = self._dict_keys
        return Compose(
            (
                LoadImaged(keys=keys),
                EnsureChannelFirstd(keys=keys),
                Spacingd(
                    keys=keys, pixdim=self.hparams.pixdim, mode=("bilinear", "nearest")
                ),
                # CropForegroundd(keys=keys, source_key="image"),
                NormalizeIntensityd(keys="image"),
                *random_transforms,
                ToTensord(keys=keys),
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
