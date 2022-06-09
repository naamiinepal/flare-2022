# from monai.data import DatasetSummary

from datamodule import DataModule

dm = DataModule(
    num_labels_with_bg=14,
    supervised_dir="/mnt/HDD2/flare2022/datasets/FLARE2022/Training/FLARE22_LabeledCase50",
    val_ratio=0.2,
    max_workers=4,
    crop_num_samples=4,
    batch_size=8,
    ds_cache_type=None,
    pixdim=(2.5, 2.5, 2.5),
    roi_size=(128, 128, 64),
)

dm.setup("fit")

# first_train = dm.train_ds[0]
first_val = dm.val_ds[0]

# dsum = DatasetSummary(dm.val_ds, num_workers=dm.num_workers)

# spacing = dsum.get_target_spacing()

# print(spacing)

# (0.7958985, 0.7958985, 2.5)
