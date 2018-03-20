"""Data processing tools."""
import numpy as np
import netCDF4
from functools import partial
from paths import data_dir
from scipy.interpolate import interp2d


def bicubic_rescale(data, scale):
    """ Rescale HR data to LR data with bicubic interpolation """
    size_HR = (np.array(data.shape)).astype(int)
    lon_HR = np.linspace(-180,-100,size_HR[0])
    lat_HR = np.linspace(-15,15,size_HR[1])
    int_func = interp2d(lat_HR, lon_HR, data, kind='cubic')
    size_LR = (np.array(data.shape) * scale).astype(int)
    lon_LR = np.linspace(-180,-100,size_LR[0])
    lat_LR = np.linspace(-15,15,size_LR[1])
    data = int_func(lat_LR, lon_LR)
    return data


def modcrop(data, scale):
    """ Adjust data to scale """
    size = data.shape
    size -= np.mod(size,scale)
    return data[:size[0],:size[1]]


def load_set(name, lr_sub_size=11, lr_sub_stride=5, scale=3):
    """ Create entire set of sub-grid from set on whole grid """
    # get HR dim from LR dim and scale
    hr_sub_size = lr_sub_size * scale
    hr_sub_stride = lr_sub_stride * scale
    # Function to generate sub-grids from entire grid
    lr_gen_sub = partial(generate_sub_grid, size=lr_sub_size,
                         stride=lr_sub_stride)
    hr_gen_sub = partial(generate_sub_grid, size=hr_sub_size,
                         stride=hr_sub_stride)
    # get sub grid
    lr_sub_arrays = []
    hr_sub_arrays = []
    # loop over all data
    for path in (data_dir / name).glob('*'):
        # load entire grid
        lr_data, hr_data = load_data_pair(str(path), scale=scale)
        # get subgrid
        lr_sub_arrays += [subgrid.reshape((subgrid.shape[0], subgrid.shape[1], 1)) for subgrid in lr_gen_sub(lr_data)]
        hr_sub_arrays += [subgrid.reshape((subgrid.shape[0], subgrid.shape[1], 1)) for subgrid in hr_gen_sub(hr_data)]
    x = np.stack(lr_sub_arrays)
    y = np.stack(hr_sub_arrays)
    return x, y


def load_data_pair(path, scale=3):
    """ Load entire HR grid and bicubic-interpolated LR grid """
    # Load netcdf file to get array
    if scale==3:
        var="SSH"
    elif scale==5:
        var="SST"
    nc = netCDF4.Dataset(path, "r")
    data = nc.variables[var][:]
    nc.close()
    # crop array to adapt to scale
    hr_data = modcrop(data, scale)
    # get low resolution grid with bicubic interpolation
    lr_data = bicubic_rescale(hr_data, 1 / scale)
    return lr_data, hr_data


def load_data_single(path, scale=3):
    """ Load entire LR grid from observations """
    if scale==3:
        var="SSH"
    elif scale==5:
        var="SST"
    nc = netCDF4.Dataset(path, "r")
    data = nc.variables[var][:]
    nc.close()
    return data


def generate_sub_grid(data, size, stride):
    """ crop whole grid to get sub-grid """
    for i in range(0, data.shape[0] - size + 1, stride):
        for j in range(0, data.shape[1] - size + 1, stride):
            yield data[i:i+size, j:j+size]

