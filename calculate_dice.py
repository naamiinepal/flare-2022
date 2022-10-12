import torch
import os
from glob import glob
import nibabel as nib
from argparse import ArgumentParser
import numpy as np

import numpy as np
import nibabel as nb
import os
from collections import OrderedDict
import pandas as pd

join = os.path.join

from monai.metrics import DiceMetric

val_dice_metric = DiceMetric(include_background=False, reduction="mean_batch")


def compute_dice_coefficient(mask_gt, mask_pred):
    """Compute soerensen-dice coefficient.
    compute the soerensen-dice coefficient between the ground truth mask `mask_gt`
    and the predicted mask `mask_pred`.

    Args:
      mask_gt: 3-dim Numpy array of type bool. The ground truth mask.
      mask_pred: 3-dim Numpy array of type bool. The predicted mask.
    Returns:
      the dice coeffcient as float. If both masks are empty, the result is NaN
    """
    volume_sum = mask_gt.sum() + mask_pred.sum()
    if volume_sum == 0:
        return np.NaN
    volume_intersect = (mask_gt & mask_pred).sum()
    return 2 * volume_intersect / volume_sum


# def calculate_dice_metric(y_pred, y) -> float:
#     """
#     Args:
#         y_pred: input data to compute, typical segmentation model output.
#         y: ground truth to compute mean dice metric.

#     Raises:
#         ValueError: when `y` is not a binarized tensor.
#         ValueError: when `y_pred` has less than three dimensions.

#     """
#     # compute dice (BxC) for each channel for each batch
#     return val_dice_metric(y_pred=y_pred, y=y)


save_name = "DSC_NSD_teamname.csv"
save_path = "~/"


seg_metrics = OrderedDict()
seg_metrics["Name"] = list()
label_tolerance = OrderedDict(
    {
        "Liver": 5,
        "RK": 3,
        "Spleen": 3,
        "Pancreas": 5,
        "Aorta": 2,
        "IVC": 2,
        "RAG": 2,
        "LAG": 2,
        "Gallbladder": 2,
        "Esophagus": 3,
        "Stomach": 5,
        "Duodenum": 7,
        "LK": 3,
    }
)
for organ in label_tolerance.keys():
    seg_metrics["{}_DSC".format(organ)] = list()


def find_lower_upper_zbound(organ_mask):
    """
    Parameters
    ----------
    seg : TYPE
        DESCRIPTION.
    Returns
    -------
    z_lower: lower bound in z axis: int
    z_upper: upper bound in z axis: int
    """
    organ_mask = np.uint8(organ_mask)
    assert np.max(organ_mask) == 1, print("mask label error!")
    z_index = np.where(organ_mask > 0)[2]
    z_lower = np.min(z_index)
    z_upper = np.max(z_index)

    return z_lower, z_upper


def main(params):
    pred_paths = glob(os.path.join(params.pred_dir, "*.nii.gz"))
    pred_paths.sort()
    gt_paths = glob(os.path.join(params.gt_dir, "*.nii.gz"))
    gt_paths.sort()

    for pred_path, gt_path in zip(pred_paths, gt_paths):

        # load grond truth and segmentation
        gt_nii = nb.load(gt_path)
        gt_data = np.uint8(gt_nii.get_fdata())
        seg_data = np.uint8(nb.load(pred_path).get_fdata())
        name = gt_nii.get_filename().split("/")[-1].split(".")[0]
        seg_metrics["Name"].append(name)

        for i, organ in enumerate(label_tolerance.keys(), 1):
            if np.sum(gt_data == i) == 0 and np.sum(seg_data == i) == 0:
                DSC_i = 1
            elif np.sum(gt_data == i) == 0 and np.sum(seg_data == i) > 0:
                DSC_i = 0
            else:
                if (
                    i == 5 or i == 6 or i == 10
                ):  # for Aorta, IVC, and Esophagus, only evaluate the labelled slices in ground truth
                    z_lower, z_upper = find_lower_upper_zbound(gt_data == i)
                    organ_i_gt, organ_i_seg = (
                        gt_data[:, :, z_lower:z_upper] == i,
                        seg_data[:, :, z_lower:z_upper] == i,
                    )
                else:
                    organ_i_gt, organ_i_seg = gt_data == i, seg_data == i
                DSC_i = compute_dice_coefficient(organ_i_gt, organ_i_seg)
            seg_metrics["{}_DSC".format(organ)].append(round(DSC_i, 4))
            print(name, organ, round(DSC_i, 4), "tol:", label_tolerance[organ])

    dataframe = pd.DataFrame(seg_metrics)
    dataframe.to_csv(join(save_path, save_name), index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--pred_dir", default="pred_dir", type=str)
    parser.add_argument("--gt_dir", default="gt_dir", type=str)

    args = parser.parse_args()
    main(args)
