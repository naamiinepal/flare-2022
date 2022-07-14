import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

# taken from https://github.com/InsightSoftwareConsortium/SimpleITK-Notebooks/blob/master/Python/gui.py
def multi_image_display2D(
    image_list,
    title_list=None,
    window_level_list=None,
    figure_size=(10, 8),
    horizontal=True,
):

    if title_list:
        if len(image_list) != len(title_list):
            raise ValueError("Title list and image list lengths do not match")
    else:
        title_list = [""] * len(image_list)

    # Create a figure.
    col_num, row_num = (len(image_list), 1) if horizontal else (1, len(image_list))
    fig, axes = plt.subplots(row_num, col_num, figsize=figure_size)
    if len(image_list) == 1:
        axes = [axes]

    # Get images as numpy arrays for display and the window level settings
    npa_list = list(map(sitk.GetArrayViewFromImage, image_list))
    if not window_level_list:
        min_intensity_list = list(map(np.min, npa_list))
        max_intensity_list = list(map(np.max, npa_list))
    else:
        min_intensity_list = list(map(lambda x: x[1] - x[0] / 2.0, window_level_list))
        max_intensity_list = list(map(lambda x: x[1] + x[0] / 2.0, window_level_list))

    # Draw the image(s)
    for ax, npa, title, min_intensity, max_intensity in zip(
        axes, npa_list, title_list, min_intensity_list, max_intensity_list
    ):
        ax.imshow(npa, cmap=plt.cm.Greys_r, vmin=min_intensity, vmax=max_intensity)
        ax.set_title(title)
        ax.set_axis_off()
    fig.tight_layout()
    return (fig, axes)