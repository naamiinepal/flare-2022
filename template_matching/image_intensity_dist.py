import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

IDX_TO_LABEL = [
    "Background",
    "Liver",
    "Right_Kidney",
    "Spleen",
    "Pancreas",
    "Aorta",
    "Inferior_venacava",
    "Right_adrenal_gland",
    "Left_adrenal_gland",
    "Gallbladder",
    "Esophagus",
    "Stomach",
    "Duodenum",
    "Left_Kidney",
]

if __name__ == "__main__":
    # moving_image_filename = '/mnt/HDD2/flare2022/datasets/FLARE2022/Training/Unlabeled/Case_01136_0000.nii.gz'
    moving_image_filename = "/mnt/HDD2/flare2022/datasets/FLARE2022/Training/FLARE22_LabeledCase50/images/FLARE22_Tr_0045_0000.nii.gz"
    moving_label_filename = "/mnt/HDD2/flare2022/datasets/FLARE2022/Training/FLARE22_LabeledCase50/labels/FLARE22_Tr_0045.nii.gz"

    image_filenames = Path(moving_image_filename).parent.rglob("*.nii.gz")
    label_filenames = Path(moving_label_filename).parent.rglob("*.nii.gz")
    stats_list_of_list = [[],]*len(IDX_TO_LABEL)
    stats_list = []
    # for image_filepath, label_filepath in zip(
    #     sorted(list(image_filenames)), sorted(list(label_filenames))
    # ):
    image = sitk.ReadImage(str(moving_image_filename))
    label = sitk.ReadImage(str(moving_label_filename))

    shape_stats = sitk.LabelShapeStatisticsImageFilter()
    shape_stats.ComputeOrientedBoundingBoxOn()
    shape_stats.Execute(label)

    intensity_stats = sitk.LabelIntensityStatisticsImageFilter()
    intensity_stats.Execute(label, image)

    [stats_list.append(
        (
            IDX_TO_LABEL[organ_id],
            shape_stats.GetPhysicalSize(organ_id),
            # shape_stats.GetElongation(i),
            # shape_stats.GetFlatness(i),
            # shape_stats.GetOrientedBoundingBoxSize(i)[0],
            # shape_stats.GetOrientedBoundingBoxSize(i)[2],
            intensity_stats.GetMean(organ_id),
            intensity_stats.GetStandardDeviation(organ_id),
            intensity_stats.GetSkewness(organ_id),
        )) for organ_id in shape_stats.GetLabels()
    ]
        
    cols = [
        "Organ",
        "Volume (mm^3)",
        # "Elongation",
        # "Flatness",
        # "Oriented Bounding Box Minimum Size(mm)",
        # "Oriented Bounding Box Maximum Size(mm)",
        "Intensity Mean",
        "Intensity Standard Deviation",
        "Intensity Skewness",
    ]
    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter("./template_matching/intensity_stats.xlsx")

    # Create the pandas data frame and display descriptive statistics.

    stats = pd.DataFrame(data=stats_list, columns=cols)
    stats.to_excel(writer, float_format="%.2f")

    writer.save()
