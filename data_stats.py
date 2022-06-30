from monai.data import DatasetSummary

from datamodules.single_step_datamodule import SingleStepDataModule

dm = SingleStepDataModule(
    num_labels_with_bg=14,
    supervised_dir="/mnt/HDD2/flare2022/datasets/FLARE2022/Training/FLARE22_LabeledCase50",
    predict_dir="/mnt/HDD2/flare2022/datasets/FLARE2022/Training/FLARE22_LabeledCase50/images",
    val_ratio=0.2,
    max_workers=4,
    crop_num_samples=4,
    batch_size=8,
    ds_cache_type=None,
    pixdim=(2.5, 2.5, 2.5),
    roi_size=(128, 128, 64),
)

dm.setup("predict")

# first_train = dm.train_ds[0]
# first_val = dm.val_ds[0]

ds = dm.pred_ds

assert len(ds), "No data found"

dsum = DatasetSummary(ds, num_workers=dm.num_workers)

print("Predict Dir", dm.hparams.predict_dir)

spacing = dsum.get_target_spacing(anisotropic_threshold=9999999999)

print(spacing)

# (0.7958985, 0.7958985, 2.5)
