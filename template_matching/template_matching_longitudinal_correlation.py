#!/usr/bin/env python
# coding: utf-8


from pathlib import Path
import SimpleITK as sitk
import numpy as np

import template_matching_utils


def RegionExtractByTemplateMatching(image: sitk.Image, template_img: sitk.Image):
    """returns a portion of the "image" that closely matches the "template_img" along with the extent(bounds) of the "image" from which the portion was extracted

    Args:
        image (sitk.Image): _description_
        template_img (sitk.Image): _description_
    Returns:
        ROI Image (sitk.Image): Region-of-Interest that matches the template image
    """
    p_image, p_template = preprocess(image, template_img)
    outTx = obtain_longitudinal_translation_params(p_image, p_template)
    print(outTx.GetParameters())


def preprocess(
    image: sitk.Image,
    template_img: sitk.Image,
    CANONICAL_ORIENTATION='RAI'
):
    """bring the template image into same physical space and orientation as the source image

    - zero-out origins
    - reorient images and template into CANONICAL orientation 
    - match spacing (change the spacing of the template image)
        itk image registration already takes this into account, so optional?
    - use projections if required


    Args:
        image (sitk.Image): _description_
        template_img (sitk.Image): _description_
    """
    # reset origin
    new_origin = (0,) * image.GetDimension()
    image.SetOrigin(new_origin)
    template_img.SetOrigin(new_origin)

    # change template orientation
    image = sitk.DICOMOrient(image,CANONICAL_ORIENTATION)
    template_img = sitk.DICOMOrient(template_img, CANONICAL_ORIENTATION)

    # change spacing of template to match the target image
    template_img = template_matching_utils.match_spacing(image, template_img)

    image_proj = template_matching_utils.generate_projections(image)
    template_proj = template_matching_utils.generate_projections(template_img)

    SAGITTAL_DIM = template_matching_utils.get_saggital_dim(CANONICAL_ORIENTATION)

    return image_proj[SAGITTAL_DIM], template_proj[SAGITTAL_DIM]


def obtain_longitudinal_translation_params(
    image: sitk.Image, template_img: sitk.Image,
    GRID_SAMPLING=20,
    DEBUG_PLOT=True
):
    """return translation transformation parameters

    Args:
        image (sitk.Image): either a AXIAL or SAGITTAL slice
        template_img (sitk.Image): same
        GRID_SAMPLING (int): step size in pixel units for the sliding window shift along longitudinal dimension
    """
    assert (
        image.GetDimension() == 2 and template_img.GetDimension()
    ), f"Images should be 2D but got {image.GetDimension()}"
    xs, YS = image.GetSize()
    xt, YT = template_img.GetSize()

    if YT >= YS:
        # template is bigger than source, do not need to traverse the longitudinal dimension
        outTx = sitk.TranslationTransform(2, (0, 0))
        return outTx

    R = sitk.ImageRegistrationMethod()
    R.SetOptimizerScales((1, 1))
    R.SetInterpolator(sitk.sitkLinear)
    R.SetMetricAsCorrelation()

    def command_iteration(method):
        if method.GetOptimizerIteration() == 0:
            print("Scales: ", method.GetOptimizerScales())
        print(
            "{0:3} = {1:7.5f} : {2}".format(
                method.GetOptimizerIteration(),
                method.GetMetricValue(),
                method.GetOptimizerPosition(),
            )
        )

    R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))

    # slide the template image through the image height(longitudinal dim)
    # until we go past the image boundary
    # generate translation parameters from 0 to image height until template overlap does not go out of bounds
    # YS is the longitudinal dimension of image
    # YT is the longitudinal dimension of template
    # we traverse in step size GRID_SAMPLING from 0 to YS(minus the template height)
    
    # generate a grid of transformations
    transform_list = [(0, index) for index in range(0, YS - YT, GRID_SAMPLING)]
    correlations_history = []
    # obtain correlation metric for each transform
    for translation_param in transform_list:
        R.SetInitialTransform(sitk.TranslationTransform(2, translation_param))
        metric = -R.MetricEvaluate(image, template_img)
        correlations_history.append(metric)
    optimal_translation_param = sitk.TranslationTransform(
        2, transform_list[np.argmax(correlations_history)]
    )
    if DEBUG_PLOT:
        # plot the template image and the target image along with the bounding box depicting
        # the location of template image in the target image
        # also plot correlation metric for each transformation
        print(YS,YT,YS-YT,transform_list)
        [print(index[-1],corr) for index,corr in zip(transform_list,correlations_history)] 
        template_matching_utils.debug_plot_template_matching(image,template_img,
        optimal_translation_param,transform_list,correlations_history)
    return optimal_translation_param


if __name__ == "__main__":

    DEFAULT_ORIENTATION = "RAI"
    source_image_filename = (
        # "/mnt/HDD2/flare2022/datasets/FLARE2022/Validation/FLARETs_0050_0000.nii.gz"
        "/mnt/HDD2/flare2022/datasets/FLARE2022/Training/Unlabeled/Case_00471_0000.nii.gz"
    )

    for fixed_image_filename in Path(source_image_filename).parent.rglob(
        "*_0000.nii.gz"
    ):

        fixed_image_filename = str(fixed_image_filename)
        moving_image_filename = "/mnt/HDD2/flare2022/datasets/FLARE2022/Training/FLARE22_LabeledCase50/images/FLARE22_Tr_0045_0000.nii.gz"

        # # Load Data
        try:
            fixed_image = sitk.ReadImage(fixed_image_filename, sitk.sitkInt16)
            moving_image = sitk.ReadImage(moving_image_filename, sitk.sitkInt16)

            outTx = RegionExtractByTemplateMatching(fixed_image, moving_image)
        except RuntimeError:
            pass
