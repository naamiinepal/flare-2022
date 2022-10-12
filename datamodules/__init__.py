import math
import os
import torch
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
    RandRotated,
    RandScaleIntensity,
    RandSpatialCropSamplesd,
    RandZoomd,
    SpatialPadd,
)
from torch.utils.data import ConcatDataset

from custom_transforms import SimulateLowResolution

from BaseSeg.data.dataset import DataLoaderX, SegDataSet
from FlareSeg.coarse_base_seg.run import get_configs

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
        num_labels_with_bg: int = 14,
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

        # self.num_workers = min(os.cpu_count(), max_workers)

        self.cache_dir = (
            f"{os.path.basename(supervised_dir)}_"
            f"{self.__class__.__name__}_datacache"
        )
        self.cfg = get_configs()
        self.num_worker = self.cfg.DATA_LOADER.NUM_WORKER
        if self.cfg.DATA_LOADER.NUM_WORKER <= self.cfg.DATA_LOADER.BATCH_SIZE + 2:
            self.num_worker = self.cfg.DATA_LOADER.BATCH_SIZE + 2

    def setup(self, stage: Optional[str] = None):
        if stage is None or stage == "fit" or stage == "validate":
            # from sklearn.model_selection import train_test_split

            # image_paths = self.get_supervised_image_paths("images")
            # label_paths = self.get_supervised_image_paths("labels")

            # data_dicts = tuple(
            #     {"image": img, "label": lab}
            #     for img, lab in zip(image_paths, label_paths)
            # )

            # train_files: List[dict]
            # val_files: List[dict]

            # train_files, val_files = train_test_split(
            #     data_dicts, test_size=self.hparams.val_ratio
            # )

            if stage != "validate":
                # train_transforms = self.get_transform(do_random=True)
                self.train_ds = self.get_dataset(stage=stage)

                # if self.hparams.do_semi:
                #     unlabeled_image_paths = glob(
                #         os.path.join(self.hparams.semisupervised_dir, "*.nii.gz")
                #     )
                #     if isinstance(self.hparams.semi_mu, int):
                #         # Sort by file size
                #         unlabeled_image_paths.sort(key=lambda img: os.stat(img).st_size)

                #         # Take only those images with smallest sizes
                #         unlabeled_image_paths = unlabeled_image_paths[
                #             : len(train_files) * self.hparams.semi_mu
                #         ]

                #     # unlabeled_files = tuple(
                #     #     {"image": img} for img in unlabeled_image_paths
                #     # )

                #     # unlabeled_transform = self.get_transform(
                #     #     keys="image", do_random=True
                #     # )

                #     unlabeled_ds = self.get_dataset(stage=stage)

                #     self.train_ds = ConcatDataset((self.train_ds, unlabeled_ds))

            # val_transforms = self.get_transform()

            self.val_ds = self.get_dataset(stage=stage)
            # if self.is_distributed_train:
            #     self.train_sampler = torch.utils.data.distributed.DistributedSampler(
            #         self.train_ds
            #     )
            #     self.val_sampler = torch.utils.data.distributed.DistributedSampler(
            #         self.val_ds
            #     )
            # else:
            #     self.train_sampler = None
            #     self.val_sampler = None
            self.train_sampler = None
            self.val_sampler = None

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
                channel_dim=0,
            )
        # # NOTE: This is a temp workaround to view validation results, delete later
        # from saver import NiftiSaver

        # self.saver = NiftiSaver(
        #     self.hparams.output_dir,
        #     output_postfix="",
        #     mode="nearest",
        #     dtype=np.float32,
        #     output_dtype=np.uint8,
        #     separate_folder=False,
        #     print_log=False,
        # )

    def train_dataloader(self):
        return DataLoaderX(
            dataset=self.train_ds,
            batch_size=self.cfg.DATA_LOADER.BATCH_SIZE,
            num_workers=self.num_worker,
            shuffle=True if self.train_sampler is None else False,
            drop_last=False,
            pin_memory=self.hparams.pin_memory,
            sampler=self.train_sampler,
        )
        # return DataLoader(
        #     self.train_ds,
        #     batch_size=math.ceil(
        #         self.hparams.batch_size / self.hparams.crop_num_samples
        #     ),
        #     num_workers=self.num_workers,
        #     shuffle=True,
        #     pin_memory=self.hparams.pin_memory,
        # )

    def val_dataloader(self):
        return DataLoaderX(
            dataset=self.val_ds,
            batch_size=self.cfg.DATA_LOADER.BATCH_SIZE,
            num_workers=self.num_worker,
            shuffle=False,
            drop_last=False,
            pin_memory=self.hparams.pin_memory,
            sampler=self.val_sampler,
        )
        # return DataLoader(
        #     self.val_ds,
        #     batch_size=1,  # Because the images do not align and are not cropped
        #     num_workers=self.num_workers,
        #     pin_memory=self.hparams.pin_memory,
        # )

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

    def get_weak_aug(
        self, keys: TupleStr, mode: TupleStr, zoom_mode: Optional[TupleStr] = None
    ):
        if zoom_mode is None:
            zoom_mode = mode

        rot_angle = math.pi / 12

        return (
            RandRotated(
                keys=keys,
                range_x=rot_angle,
                range_y=rot_angle,
                range_z=rot_angle,
                dtype=np.float32,
                padding_mode="zeros",
                mode=mode,
                prob=0.9,
            ),
            RandZoomd(
                keys=keys,
                min_zoom=0.8,
                max_zoom=1.3,
                mode=zoom_mode,
                prob=0.9,
                padding_mode="constant",
                keep_size=False,  # Last spatial padd will handle the case
            ),
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

    def get_dataset(self, stage="train"):
        return SegDataSet(self.cfg, stage)
