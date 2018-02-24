import numpy as np
import netCDF4
from functools import partial
from toolbox.image import bicubic_rescale
from toolbox.image import modcrop
from toolbox.paths import data_dir


def load_set(name, lr_sub_size=11, lr_sub_stride=5, scale=3):
    """ Create entire set of sub-images from entire set of whole image """
    # get HR dim from LR dim and scale
    hr_sub_size = lr_sub_size * scale
    hr_sub_stride = lr_sub_stride * scale
    # Function to generate sub-images from entire image
    lr_gen_sub = partial(generate_sub_images, size=lr_sub_size,
                         stride=lr_sub_stride)
    hr_gen_sub = partial(generate_sub_images, size=hr_sub_size,
                         stride=hr_sub_stride)
    # get sub images
    lr_sub_arrays = []
    hr_sub_arrays = []
    # loop over all images
    for path in (data_dir / name).glob('*'):
        # load entire image
        lr_image, hr_image = load_image_pair(str(path), scale=scale)
        # get subimages
        lr_sub_arrays += [img.reshape((img.shape[0], img.shape[1], 1)) for img in lr_gen_sub(lr_image)]
        hr_sub_arrays += [img.reshape((img.shape[0], img.shape[1], 1)) for img in hr_gen_sub(hr_image)]
    x = np.stack(lr_sub_arrays)
    y = np.stack(hr_sub_arrays)
    return x, y


def load_image_pair(path, scale=3):
    """ Load entire single cropped HR and bicubic-interpolated LR image """
    # Load netcdf file to get array
    nc = netCDF4.Dataset(path, "r")
    image = nc.variables["SSH"][:] 
    nc.close()
    # crop array to adapt to scale
    hr_image = modcrop(image, scale)
    # get low resolution image with bicubic interpolation
    lr_image = bicubic_rescale(hr_image, 1 / scale)
    return lr_image, hr_image


def generate_sub_images(image, size, stride):
    """ crop whole image to get sub-images """
    for i in range(0, image.shape[0] - size + 1, stride):
        for j in range(0, image.shape[1] - size + 1, stride):
            yield image[i:i+size, j:j+size] 

