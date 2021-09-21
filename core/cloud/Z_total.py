from netCDF4 import Dataset
import numpy as np
from os import mkdir, chdir, getcwd

from .Z_graupel import Z_graupel
from .Z_snow import Z_snow
from .Z_cldw import Z_cldw
from .Z_rain import Z_rain
from .Z_cldi import Z_cldi


def Z_total(filename, mode=0, save_data=0):
    """
    Calculate data and save them in .npz, also return a dict which contain them
    :param save_data: if you want to save data in file, set this to 1
    :param mode: if mode is 1, the function is forced to re-calculate data.
    :param filename: nc file name
    :return: Z:dict, K:dict
    """
    work_path = getcwd()
    try:
        chdir('data')
        chdir(work_path)
    except FileNotFoundError:
        mkdir('data')
        mode = 1

    if not mode:
        try:
            f = open('data/{0}_Z.npz'.format(filename[:-3]), 'r+b')
            f.close()
            f = open('data/{0}_K.npz'.format(filename[:-3]), 'r+b')
            f.close()
            print('Z_total: Data of {0} exists, so pass it\nif you want to re-calculate it, set mode=1\n'.format(
                filename))
            return np.load('data/{0}_Z.npz'.format(filename[:-3])), np.load('data/{0}_K.npz'.format(filename[:-3]))
        except FileNotFoundError:
            pass

    Data = Dataset(filename)

    # Cldi
    RMax = 2.5
    Density = 917
    N0 = 0.1
    Mu = 1
    ZCldi, KCldi = Z_cldi(Data, RMax, Density, N0, Mu)

    # Cldw
    RMax = 0.1
    Density = 1000
    N0 = 0.02
    Theta = 0.35
    ZCldw, KCldw = Z_cldw(Data, RMax, Density, N0, Theta)

    # Snow
    RMax = 5
    Density = 100
    N0 = 1 * 10 ** 8
    ZSnow, KSnow = Z_snow(Data, RMax, Density, N0)

    # Rain
    RMax = 5
    Density = 1000
    N0 = 2.2 * 10 ** 7
    ZRain, KRain = Z_rain(Data, RMax, Density, N0)

    # Graupel
    RMax = 5
    Density = 400
    N0 = 4 * 10 ** 6
    ZGraupel, KGraupel = Z_graupel(Data, RMax, Density, N0)

    # get specific data and convert data to log10() * 10
    ZTotal = np.log10(ZCldw + ZCldi + ZSnow + ZRain + ZGraupel) * 10
    KTotal = (KCldw + KCldi + KSnow + KRain + KGraupel) / 10 ** 9

    # save all Z data and K data to one file
    if save_data:
        np.savez('data/{0}_Z.npz'.format(filename[:-3]), cldi=ZCldi, cldw=ZCldw, rain=ZRain, snow=ZSnow,
                 graupel=ZGraupel, total=ZTotal)
        np.savez('data/{0}_K.npz'.format(filename[:-3]), cldi=KCldi, cldw=KCldw, rain=KRain, snow=KSnow,
                 graupel=KGraupel, total=KTotal)
        print('All the Z* and K* data has been saved in "data" folder')

    print('Finish!')

    return {
               'total': ZTotal,
               'cldi': ZCldi,
               'cldw': ZCldw,
               'rain': ZRain,
               'snow': ZSnow,
               'graupel': ZGraupel}, {
               'total': KTotal,
               'cldi': KCldi,
               'cldw': KCldw,
               'rain': KRain,
               'snow': KSnow,
               'graupel': KGraupel
           }
