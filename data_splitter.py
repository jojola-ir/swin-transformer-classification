'''Dataset splitter'''
import argparse
import os
from glob import iglob

import imageio.core.util
import nibabel as nib
import numpy as np
import splitfolders
from skimage import io

SLICE_X = True
SLICE_Y = True
SLICE_Z = True

SLICE_DECIMATE_IDENTIFIER = 3


def silence_imageio_warning(*args, **kwargs):
    pass


imageio.core.util._precision_warn = silence_imageio_warning


def saveSlice(img, fname, path):
    # img = np.uint8(img * 255)
    fout = os.path.join(path, f'{fname}.png')
    io.imsave(fout, img, check_contrast=False)
    print(f'[+] Slice saved: {fout}', end='\r')


def sliceAndSaveVolumeImage(vol, fname, path):
    (dimx, dimy, dimz) = vol.shape
    print(dimx, dimy, dimz)
    cnt = 0
    if SLICE_X:
        cnt += dimx
        print('Slicing X: ')
        for i in range(dimx):
            saveSlice(vol[i, :, :], fname + f'-slice{str(i).zfill(SLICE_DECIMATE_IDENTIFIER)}_x', path)

    if SLICE_Y:
        cnt += dimy
        print('Slicing Y: ')
        for i in range(dimy):
            saveSlice(vol[:, i, :], fname + f'-slice{str(i).zfill(SLICE_DECIMATE_IDENTIFIER)}_y', path)

    if SLICE_Z:
        cnt += dimz
        print('Slicing Z: ')
        for i in range(dimz):
            saveSlice(vol[:, :, i], fname + f'-slice{str(i).zfill(SLICE_DECIMATE_IDENTIFIER)}_z', path)
    return cnt


def generate_dataset(path_to_images, path_to_output, organ="heart", test=False):
    """Generates .png images from .nii images.
    Parameters
    ----------
    path_to_images : str
        Path to the .nii images.
    path_to_output : str
        Output path for .png images.
    organ : str
        Organ name.
    test : boolean
        Enables or disables testset generation.
    Returns
    -------
    None
    """

    if test:
        imagePathInput = os.path.join(path_to_images, 'imagesTs/')

        imageSliceOutput = os.path.join(path_to_output, 'images/')

        for root, _, files in os.walk(imagePathInput):
            for f in files:
                print(f)

        if os.path.exists(imageSliceOutput) is False:
            os.makedirs(imageSliceOutput)

        for index, filename in enumerate(sorted(iglob(imagePathInput + '*.nii.gz'))):
            img = nib.load(filename).get_fdata()
            print(filename, img.shape, np.sum(img.shape), np.min(img), np.max(img))
            numOfSlices = sliceAndSaveVolumeImage(img, organ + str(index), imageSliceOutput)
            print(f'\n{filename}, {numOfSlices} slices created \n')

    else:
        imagePathInput = os.path.join(path_to_images, 'imagesTr/')
        maskPathInput = os.path.join(path_to_images, 'labelsTr/')

        imageSliceOutput = os.path.join(path_to_output, 'images/')
        maskSliceOutput = os.path.join(path_to_output, 'masks/')

        if os.path.exists(imageSliceOutput) is False:
            os.makedirs(imageSliceOutput)

        if os.path.exists(maskSliceOutput) is False:
            os.makedirs(maskSliceOutput)

        for index, filename in enumerate(sorted(iglob(imagePathInput + '*.nii.gz'))):
            img = nib.load(filename).get_fdata()
            print(filename, img.shape, np.sum(img.shape), np.min(img), np.max(img))
            numOfSlices = sliceAndSaveVolumeImage(img, organ + str(index), imageSliceOutput)
            print(f'\n{filename}, {numOfSlices} slices created \n')

        for index, filename in enumerate(sorted(iglob(maskPathInput + '*.nii.gz'))):
            img = nib.load(filename).get_fdata()
            print(filename, img.shape, np.sum(img.shape), np.min(img), np.max(img))
            numOfSlices = sliceAndSaveVolumeImage(img, organ + str(index), maskSliceOutput)
            print(f'\n{filename}, {numOfSlices} slices created \n')


def random_splitter(src, dest, test_rate, clear):
    """Splits generated images (slices) into train/val/test subdirectories.
        Parameters
        ----------
        src : str
            Path to the slices directory.
        dest : str
            Path to the destination directory.
        test_rate : float
            Test images rate.
        clear : boolean
            Enables or disables useless images clearing.
        """

    assert test_rate < 1, "test_rate must be less than 1"

    if clear:
        k = 0
        for root, _, files in os.walk(src):
            if root == os.path.join(src, "masks"):
                for f in files:
                    if not f.endswith(".DS_Store"):
                        maskpath = os.path.join(root, f)
                        imagepath = os.path.join(os.path.join(src, "images/"), f)
                        if os.lstat(maskpath).st_size <= 120:  # set file size in kb
                            k += 1
                            print(f)
                            os.remove(imagepath)
                            os.remove(maskpath)
        print(f"Images found : {k}")

    val_rate = 0.1

    splitfolders.ratio(src, output=dest,
                       seed=1337, ratio=(1 - (test_rate + val_rate), val_rate, test_rate), group_prefix=None,
                       move=False)


def main():
    """This is the main function"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", "-g", default=False, help="generate slices",
                        action="store_true")
    parser.add_argument("--datapath", help="path to the dataset")
    parser.add_argument("--output", "-o", help="path to the output", default="data/")
    parser.add_argument("--test", help="Generate test set", default=False, action="store_true")
    parser.add_argument("--testrate", "-t", help="part of test set", default=0.2)
    parser.add_argument("--clear", "-c", help="clear useless images", default=False, action="store_true")

    args = parser.parse_args()

    generate = args.generate
    datapath = args.datapath
    test = args.test
    output = args.output

    test_rate = args.testrate
    clear = args.clear

    if generate:
        output = args.output
        generate_dataset(datapath, output, test=test)

    random_splitter(output, output, test_rate, clear)


if __name__ == "__main__":
    main()
