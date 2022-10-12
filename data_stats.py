from glob import glob
import os
import nibabel as nib
import numpy as np

base_dir = "/mnt/HDD2/flare2022/datasets/FLARE2022/Training/Unlabeled"


def get_image_paths(baseDir: str):
    image_paths = glob(os.path.join(baseDir, "*.nii.gz"))
    image_paths.sort()
    return image_paths


image_paths = get_image_paths(base_dir)
assert len(image_paths), "No data found"

voxel_spacing_max, voxel_spacing_min = np.array((0.0, 0.0, 0.0)), np.array(
    (100.0, 100.0, 100.0)
)

spatial_size_max, spatial_size_min = np.array((0.0, 0.0, 0.0)), np.array(
    (10000.0, 10000.0, 10000.0)
)

for image_path in image_paths:
    img = nib.load(image_path)
    spacing = img.header.get_zooms()[:3]
    size = img.shape
    print(image_path.split("/")[-1], spacing, size)
    spatial_size_max = np.maximum(spatial_size_max, size)
    spatial_size_min = np.minimum(spatial_size_min, size)
    voxel_spacing_min = np.minimum(voxel_spacing_min, spacing)
    voxel_spacing_max = np.maximum(voxel_spacing_max, spacing)


print("Voxel Spacing Min", voxel_spacing_min)
print("Voxel Spacing Max", voxel_spacing_max)

print("Spatial Size Min", spatial_size_min)
print("Spatial Size Max", spatial_size_max)
# spacing = dsum.get_target_spacing(anisotropic_threshold=9999999999)

# print(spacing)

# (0.7958985, 0.7958985, 2.5)
