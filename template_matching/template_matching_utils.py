import os
import SimpleITK as sitk
from matplotlib import image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib2 import Path

def plot_hist(image:sitk.Image):
    fig = plt.figure()
    plt.hist(sitk.GetArrayViewFromImage(image).flatten(),bins=100)
    plt.show()

def generate_projections(image:sitk.Image):
    assert image.GetDimension() > 2, 'Image should have more than 2 dimension'
    projection_list = []
    for i in range(image.GetDimension()):
        p = extract_slice_helper(sitk.MeanProjection(image,i),dim=i,index=0)
        projection_list.append(p)
    return projection_list
                              
def generate_random_str(length=5):
  alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
  return ''.join(np.random.choice(list(alphabet),length,replace=True))

def get_saggital_dim(orientation:str='LPS'):
  for i,direction in enumerate(list(orientation)):
    if direction == 'L' or direction == 'R':
      return i
  raise ValueError(f'orientation string {orientation} is invalid')

def get_axial_dim(orientation:str='LPS'):
  for i,direction in enumerate(list(orientation)):
    if direction == 'A' or direction == 'P':
      return i
  raise ValueError(f'orientation string {orientation} is invalid')

def get_longitudinal_dim(orientation:str='LPS'):
  for i,direction in enumerate(list(orientation)):
    if direction == 'S' or direction == 'I':
      return i
  raise ValueError(f'orientation string {orientation} is invalid')

def debug_plot(image_slice:sitk.Image, template_slice:sitk.Image, outTx:sitk.sitkTranslation,transform_list,correlations):

  image_slice = make_isotropic(image_slice)
  template_slice = make_isotropic(template_slice)
  
  if image_slice.GetDimension() == 3:
    axial_dim = get_saggital_dim(sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(image_slice.GetDirection()))
    axial_dim_size = image_slice.GetSize()[axial_dim]
    image_slice = extract_slice_helper(image_slice,dim=axial_dim,index=axial_dim_size//2)
  if template_slice.GetDimension() == 3:
    axial_dim = get_saggital_dim(sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(template_slice.GetDirection()))
    axial_dim_size = image_slice.GetSize()[axial_dim]
    template_slice = extract_slice_helper(template_slice,dim=axial_dim,index=axial_dim_size//2)

  fig = plt.figure(figsize=(10,10))
  ax = plt.subplot(1,3,1)
  plt.imshow(sitk.GetArrayViewFromImage(template_slice),cmap='gray')


  ax2 = plt.subplot(1,3,2)
  plt.imshow(sitk.GetArrayViewFromImage(image_slice),cmap='gray')
  # Create a Rectangle patch
  rect = patches.Rectangle(outTx.GetParameters(), *template_slice.GetSize(), linewidth=1, edgecolor='r', facecolor='none',
                          clip_on=False)

  # Add the patch to the Axes
  ax2.add_patch(rect)

  ax3 = plt.subplot(1,3,3)
  # ax3.sharey(ax2)
  ax3.plot(np.array([tx[-1] for tx in transform_list]),correlations)

  plt.tight_layout()
  out_filepath = 'template_matching/images/'+generate_random_str()+'.png' 
  os.makedirs(str(Path(out_filepath).parent),exist_ok=True)
  print(f'Saving debug plot to {out_filepath}')
  plt.savefig(out_filepath)
  # plt.show()  
  
def extract_slice_helper(image:sitk.Image,dim=0,index=0):
    size = list(image.GetSize())
    size[dim] = 0
    slice_index = [0,]*len(size)
    slice_index[dim] = index
    return sitk.Extract(image,size,slice_index)

def make_isotropic(image: sitk.Image,is_label= False):
    """return a new image whose voxel dimension is isotropic"""
    ts = (1,)*image.GetDimension()
    im_sp = image.GetSpacing()
    new_spacing = ts
    im_sz = image.GetSize()

    # expand or contract the image size as required
    out_sz = []
    for i in range(image.GetDimension()):
        out_sz.append(
            int(np.round(im_sz[i] * (im_sp[i] / new_spacing[i]))),
        )

    resample = sitk.ResampleImageFilter()
    # make the output spacing isotropic

    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(out_sz)  # newly calculated size
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(image.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkLinear)

    return resample.Execute(image)

def match_spacing(reference: sitk.Image, image: sitk.Image, is_label=False):
    """return a new image (with content same as image)
    which matches the reference image spacing

    Args:
        image (sitk.Image): _description_
        reference (sitk.Image): _description_
        is_label (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    assert image.GetDimension() == reference.GetDimension(
    ), f'Image and reference image should have same dimensions, got {image.GetDimension()} and {reference.GetDimension()}'

    ts = reference.GetSpacing()
    im_sp = image.GetSpacing()
    # new_spacing = (im_sp[0],im_sp[1],ts[2])
    new_spacing = ts
    im_sz = image.GetSize()

    # expand or contract the image size as required
    out_sz = []
    for i in range(image.GetDimension()):
        out_sz.append(
            int(np.round(im_sz[i] * (im_sp[i] / new_spacing[i]))),
        )

    resample = sitk.ResampleImageFilter()
    # make the output spacing to match reference spacing
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(out_sz)  # newly calculated size
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(image.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkLinear)

    return resample.Execute(image)

if __name__ == '__main__':
  assert 0 == get_saggital_dim('LPS'), 'get_saggital_dim() failed'
  assert 0 == get_saggital_dim('RAI'), 'get_saggital_dim() failed'  
  assert 1 == get_saggital_dim('PRI'), 'get_saggital_dim() failed'  