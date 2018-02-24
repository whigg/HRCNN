"""Image processing tools."""
import numpy as np
from scipy.interpolate import interp2d


def bicubic_rescale(image, scale):
    """ Rescale HR image to LR image with bicubic interpolation """
    size_HR = (np.array(image.shape)).astype(int)
    lon_HR = np.linspace(-180,-100,size_HR[0])
    lat_HR = np.linspace(-15,15,size_HR[1])
    int_func = interp2d(lat_HR, lon_HR, image, kind='cubic')
    size_LR = (np.array(image.shape) * scale).astype(int)
    lon_LR = np.linspace(-180,-100,size_LR[0])
    lat_LR = np.linspace(-15,15,size_LR[1])
    image = int_func(lat_LR, lon_LR)
    return image


def modcrop(image, scale):
    """ Adjust image to scale """
    size = image.shape
    size -= np.mod(size,scale)
    return image[:size[0],:size[1]]

