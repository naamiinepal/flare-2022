import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import skimage
import matplotlib.animation as animation
from argparse import ArgumentParser
import glob
import os


def show_slices(img, images_3d):
    """ Function to display row of image slices """
    fig, axes = plt.subplots(len(img.shape), 1, figsize=(3, 9), dpi=100)
    for i in range(img.shape[0]):
        plt_0 = axes[0].imshow(np.rot90(img[i, :, :]), cmap="gray", animated=True, aspect='equal', interpolation='nearest')
        axes[0].axis('off')
        plt_1 = axes[1].imshow(np.rot90(img[:, i, :]), cmap="gray", animated=True, aspect='equal', interpolation='nearest')
        axes[1].axis('off')
        plt_2 = axes[2].imshow(np.rot90(img[:, :, i]), cmap="gray", animated=True, aspect='equal', interpolation='nearest')
        axes[2].axis('off')

        images_3d.append([plt_0, plt_1, plt_2])
    return fig, images_3d

def img_to_gif(img_filename, out_folder):
    images_3d = []
    filename = img_filename.split('/')[-1]
    filename = filename.split('.')[0]
    img = nib.load(img_filename).get_fdata()
    img = skimage.transform.resize(img, (256, 256, 256), mode='edge')


    fig, images_3d = show_slices(img, images_3d)
    fig.set_dpi(100)
    fig.subplots_adjust(bottom=0, top=1, left=0, right=1)
    anim = animation.ArtistAnimation(fig, images_3d)
    anim.save(os.path.join(out_folder, f'{filename}.gif'), fps=12, writer="pillow")
    print(f"{filename}.gif is done")


def main(args):
    folder = glob.glob(args.in_folder + "/*.nii.gz")
    for img_filename in folder:
        print(img_filename)
        img_to_gif(img_filename, args.out_folder)
    

if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("--in_folder", type=str, default="/mnt/HDD2/flare2022/datasets/FLARE2022/Training/FLARE22_LabeledCase50/images")
    parser.add_argument("--out_folder", type=str, default="/mnt/HDD2/flare2022/gifs/")
    args = parser.parse_args()
    main(args)
