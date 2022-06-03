import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from glob import glob

from monai.data import Dataset, DatasetSummary
from monai.transforms import LoadImaged


def get_task_params(args):
    """
    This function is used to achieve the spacings of decathlon dataset.
    In addition, for CT images (task 03, 06, 07, 08, 09 and 10), this function
    also prints the mean and std values (used for normalization), and the min (0.5 percentile)
    and max(99.5 percentile) values (used for clip).

    """
    root_dir = args.root_dir

    train_image_paths = glob(os.path.join(root_dir, "images", "*.nii.gz"))
    train_image_paths.sort()

    train_label_paths = glob(os.path.join(root_dir, "labels", "*.nii.gz"))
    train_label_paths.sort()

    data_dicts = tuple(
        {"image": img, "label": label}
        for img, label in zip(train_image_paths, train_label_paths)
    )
    # print(data_dicts)
    dataset = Dataset(
        data=data_dicts,
        transform=LoadImaged(keys=["image", "label"]),
    )

    calculator = DatasetSummary(dataset, num_workers=4)
    target_spacing = calculator.get_target_spacing()
    print("spacing: ", target_spacing)
    print("CT input, calculate statistics:")
    calculator.calculate_statistics()
    print("mean: ", calculator.data_mean, " std: ", calculator.data_std)
    calculator.calculate_percentiles(
        sampling_flag=True, interval=10, min_percentile=0.5, max_percentile=99.5
    )
    print(
        "min: ",
        calculator.data_min_percentile,
        " max: ",
        calculator.data_max_percentile,
    )


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "-root_dir",
        "--root_dir",
        type=str,
        default="/mnt/HDD2/flare2022/datasets/FLARE2022/Training/FLARE22_LabeledCase50/",
        help="dataset path",
    )

    args = parser.parse_args()
    get_task_params(args)
