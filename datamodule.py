import math
import os.path
from glob import glob
from typing import Literal, Optional, Tuple

import pytorch_lightning as pl
from monai.data import (
    CacheDataset,
    DataLoader,
    Dataset,
    PersistentDataset,
)
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    LoadImaged,
    # Orientationd,
    NormalizeIntensityd,
    RandCropByLabelClassesd,
    ToTensord,
)
from sklearn.model_selection import train_test_split

from saver import NiftiSaver


class DataModule(pl.LightningDataModule):
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
        pin_memory: bool = True,
        **kwargs
    ):
        super().__init__()

        self._dict_keys = ("image", "label")

        self.save_hyperparameters()

        self.num_workers = min(os.cpu_count(), max_workers)

    def setup(self, stage: Optional[str] = None):
        if stage is None or stage == "fit" or stage == "validate":
            images = self.get_image_paths("images")
            labels = self.get_image_paths("labels")

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

                train_transforms = self.get_transform(
                    RandCropByLabelClassesd(
                        keys=self._dict_keys,
                        label_key="label",
                        spatial_size=self.hparams.roi_size,
                        num_samples=self.hparams.crop_num_samples,
                        num_classes=self.hparams.num_labels_with_bg,
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
                self.train_ds = self.get_dataset(train_files, train_transforms)

            val_transforms = self.get_transform()

            self.val_ds = self.get_dataset(val_files, val_transforms)

        if stage is None or stage == "predict":
            pred_image_paths = glob(os.path.join(self.hparams.predict_dir, "*.nii.gz"))
            pred_image_paths.sort()

            pred_dicts = tuple({"image": img} for img in pred_image_paths)

            keys = "image"
            pred_transforms = Compose(
                (
                    LoadImaged(keys=keys),
                    EnsureChannelFirstd(keys=keys),
                    # Orientationd(keys=keys, axcodes="RAS"),
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

    def get_image_paths(self, baseDir: str):
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
                # Orientationd(keys=keys, axcodes="RAS"),
                #         Spacingd(keys=keys, pixdim=(
                #             1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
                #         ScaleIntensityRanged(
                #             "image", a_min=-57, a_max=164,
                #             b_min=0.0, b_max=1.0, clip=True,
                #         ),
                CropForegroundd(keys=keys, source_key="image"),
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
