import os.path
from glob import glob
from typing import Optional, Tuple

import pytorch_lightning as pl
from monai.data import CacheDataset, DataLoader, Dataset
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    RandCropByLabelClassesd,
    ToTensord,
)
from sklearn.model_selection import train_test_split


class FlareDataModule(pl.LightningDataModule):

    NUM_LABELS = 13

    def __init__(
        self,
        val_ratio: float = 0.2,
        crop_num_samples: int = 4,
        batch_size: int = 16,
        cache_ds: bool = True,
        max_workers: int = 4,
        roi_size: Tuple[int, int, int] = (128, 128, 64),
        **kwargs
    ):
        super().__init__()

        self._dict_keys = ("image", "label")

        data_dir = "/mnt/HDD2/flare2022/datasets/FLARE2022"
        self.supervised_dir = os.path.join(
            data_dir, "Training", "FLARE22_LabeledCase50"
        )

        self.num_workers = min(os.cpu_count(), max_workers)

        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None):
        if stage is None or stage == "fit":
            images = self.get_image_paths("images")
            labels = self.get_image_paths("labels")

            data_dicts = tuple(
                {"image": img, "label": lab} for img, lab in zip(images, labels)
            )

            train_files, val_files = train_test_split(
                data_dicts, test_size=self.hparams.val_ratio
            )

            train_transforms = self.get_transform(
                RandCropByLabelClassesd(
                    keys=self._dict_keys,
                    label_key="label",
                    spatial_size=self.hparams.roi_size,
                    num_samples=self.hparams.crop_num_samples,
                    num_classes=self.NUM_LABELS + 1,
                ),
                # user can also add other random transforms
                #                     RandAffined(
                #                         keys=keys,
                #                         mode=('bilinear', 'nearest'),
                #                         prob=1.0,
                #                         rotate_range=(0, 0, math.pi/15),
                #                         scale_range=(0.1, 0.1, 0.1)
                #                     )
            )
            val_transforms = self.get_transform()

            self.train_ds = self.get_dataset(train_files, train_transforms)

            self.val_ds = self.get_dataset(val_files, val_transforms)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size // self.hparams.crop_num_samples,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=1,  # Because the images do not align and are not cropped
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def get_image_paths(self, baseDir: str):
        image_paths = glob(os.path.join(self.supervised_dir, baseDir, "*.nii.gz"))
        image_paths.sort()
        return image_paths

    def get_transform(self, *random_transforms):
        keys = self._dict_keys
        return Compose(
            (
                LoadImaged(keys=keys),
                EnsureChannelFirstd(keys=keys),
                Orientationd(keys=keys, axcodes="RAS"),
                #         Spacingd(keys=keys, pixdim=(
                #             1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
                #         ScaleIntensityRanged(
                #             "image", a_min=-57, a_max=164,
                #             b_min=0.0, b_max=1.0, clip=True,
                #         ),
                CropForegroundd(keys=keys, source_key="image"),
                *random_transforms,
                ToTensord(keys=keys),
            )
        )

    def get_dataset(self, *dataset_args):
        return (
            CacheDataset(*dataset_args, num_workers=self.num_workers)
            if self.hparams.cache_ds
            else Dataset(*dataset_args)
        )
