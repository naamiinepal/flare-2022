from matplotlib import pyplot as plt
import numpy as np
from pathlib2 import Path
import SimpleITK as sitk

from template_matching.template_matching_utils import generate_projections, get_isotropic_size, make_isotropic
from template_matching.visualize_utils import multi_image_display2D


if __name__ == "__main__":
    moving_image_filename = "/mnt/HDD2/flare2022/datasets/FLARE2022/Training/FLARE22_LabeledCase50/images/FLARE22_Tr_0045_0000.nii.gz"
    # moving_image_filename = '/mnt/HDD2/flare2022/datasets/FLARE2022/Training/Unlabeled/Case_01136_0000.nii.gz'
    size_list = []

    for index, template_image in enumerate(
        Path(moving_image_filename).parent.rglob("*_0000.nii.gz")
    ):
        filename = Path(template_image)
        try:
            image = sitk.ReadImage(str(filename))
            size = get_isotropic_size(image)
            print(index + 1, filename.stem, size)
            size_list.append((size, filename.stem))
            iso_image = make_isotropic(image)
            iso_image = sitk.DICOMOrient(iso_image,'RAI')
            iso_proj = generate_projections(iso_image)
            multi_image_display2D(iso_proj)
            plt.savefig(f'./template_matching/images/{str(filename.stem)[:-4]}.png')
        except RuntimeError as e:
            print(index + 1, filename.stem, e)

    # stats
    size_list.sort(key=lambda x: x[0][0])
    print(
        f"Min: {size_list[0]} Max: {size_list[-1]} Median {size_list[len(size_list)//2]}"
    )

    # plot the min, median and max images in terms of size

    _, min_name = size_list[0]
    _,max_name = size_list[-1]
    _,median_name = size_list[-1]

    base_path = Path(moving_image_filename).parent
    min_image = make_isotropic(sitk.ReadImage(str(base_path/min_name)))
    max_image = make_isotropic(sitk.ReadImage(str(base_path/max_name)))
    median_image = make_isotropic(sitk.ReadImage(str(base_path/median_name)))

    min_proj = generate_projections(min_image)
    max_proj = generate_projections(max_image)
    median_proj = generate_projections(median_image)


    # turns out all labeled examples are in RAS orientation
    # the Unlabelled examples can be in wildly varying orientation including LPS, RAS, LAI,
    multi_image_display2D([min_proj[0],max_proj[0],median_proj[0]])
    plt.savefig('./template_matching/images/template_images.png')