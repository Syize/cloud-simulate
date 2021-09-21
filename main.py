#!/usr/bin/python3

from time import perf_counter
import numpy as np
from core import Z_total, RadarCoupling
from core.tools import draw_cloud, draw_radar


def get_data(nc, name, level=None, axis=None):
    """
    This function is used to get data in nc file
    :param nc:
    :param name:
    :param level:
    :param axis:
    :return:
    """
    if level is not None:
        if axis == 1:
            return np.asarray(nc[name])[0][:, level, :]
        elif axis == 0 or axis is None:
            return np.asarray(nc[name])[0][:, :, level]
        else:
            raise Exception('axis should be 0, 1 or None')
    else:
        raise Exception('You must give level to get specific layer of data')


def read_npz(filename):
    """
    read data from .npz
    :param filename:
    :return:
    """
    r = np.load(filename)
    return r['Z'], r['K']


def get_Z_data(data, level, axis):
    """
    This function is used to get data in ReturnDict
    :param data:
    :param level:
    :param axis:
    :return:
    """
    if axis == 1:
        return data[:, level, :]
    else:
        return data[:, :, level]


def run(filename):
    start = perf_counter()

    # filename = 'wrfout_d03_2016-06-23_04_30_00.nc'

    Z, K = Z_total(filename)
    sj_res = draw_cloud(filename, Z, K, 0, 199)
    dbz, pia = RadarCoupling(filename, Z['total'], sj_res['total'])
    draw_radar(filename, dbz, pia, axis=0, level=199)

    print('Finish! Time usage:', perf_counter() - start, 's')


if __name__ == '__main__':
    run()
