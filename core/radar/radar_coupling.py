# -*- coding:utf-8 -*-
from rich.progress import Progress
import numpy as np
from os import chdir, getcwd, mkdir

from ..tools import Gaussian, ConvertArray


def RadarCoupling(filename, z_total, sj_total, mode=0, save_data=0):
    """
    radar coupling
    :param mode:
    :param save_data:
    :param filename: nc file name
    :param z_total: Z total data
    :param sj_total: attenuation total data
    :return:
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
            f = open('data/{0}_coupling.npz'.format(filename[:-3]), 'r+b')
            f.close()
            print('{0}_coupling.npz exists, so pass it\nif you want to re-calculate it, set mode=1\n'.format(
                filename[:-3]))
            return np.load('data/{0}_coupling.npz'.format(filename[:-3]))
        except FileNotFoundError:
            pass

    # z, y, x = z_total.shape
    FWHMx = 1.4
    FWHMy = 1.7
    # level = 30
    dx = 0.5
    dy = 0.5
    x_sub = round(FWHMx / dx)
    y_sub = round(FWHMy / dy)

    # ################################# TEST CODE ###############################
    # test = np.array([
    #     [1, 2, 3, 4],
    #     [5, 6, 7, 8],
    #     [9, 10, 11, 12],
    #     [13, 14, 15, 16]
    # ])
    # print(ConvertArray(-2, 0, test))
    # ###########################################################################

    # 开始卷，定义向右为正方向
    with Progress() as process:
        pid = process.add_task("[red]Radar coupling is running...", total=50)
        dbz_out = []
        PIA_out = []
        # z_num = z_total.shape[0]
        for dbz, sj in zip(z_total, sj_total):
            dbz = np.power(10, dbz / 10)
            # sj = np.power(10, -0.1 * sj)
            radar_dbz_out = dbz * dx * dy
            radar_PIA_out = sj * dx * dy
            normalize = dx * dy
            # 0要单独计算
            for ix in range(1, x_sub + 1):
                for iy in range(1, y_sub + 1):
                    # ix is positive
                    dx_new = ix * dx
                    # # iy is negative
                    dy_new = -iy * dy
                    # # 想整张矩阵计算就要对矩阵做变换，将对应的点平移到高斯卷积的点上
                    radar_dbz_out = radar_dbz_out + Gaussian(dx_new / FWHMx, dy_new / FWHMy) * ConvertArray(
                        ix, -iy,
                        dbz)
                    radar_PIA_out = radar_PIA_out + Gaussian(dx_new / FWHMx, dy_new / FWHMy) * ConvertArray(
                        ix, -iy,
                        sj)
                    normalize = normalize + Gaussian(dx_new / FWHMx, dy_new / FWHMy)

                    # # iy is positive
                    dy_new = iy * dy
                    radar_dbz_out = radar_dbz_out + Gaussian(dx_new / FWHMx, dy_new / FWHMy) * ConvertArray(
                        ix, iy,
                        dbz)
                    radar_PIA_out = radar_PIA_out + Gaussian(dx_new / FWHMx, dy_new / FWHMy) * ConvertArray(
                        ix, iy,
                        sj)
                    normalize = normalize + Gaussian(dx_new / FWHMx, dy_new / FWHMy)

                    # ix is negative
                    dx_new = -ix * dx
                    # # iy is negative
                    dy_new = -iy * dy
                    radar_dbz_out = radar_dbz_out + Gaussian(dx_new / FWHMx, dy_new / FWHMy) * ConvertArray(
                        -ix, -iy,
                        dbz)
                    radar_PIA_out = radar_PIA_out + Gaussian(dx_new / FWHMx, dy_new / FWHMy) * ConvertArray(
                        -ix, -iy,
                        sj)
                    normalize = normalize + Gaussian(dx_new / FWHMx, dy_new / FWHMy)
                    # # iy is positive
                    dy_new = iy * dy
                    radar_dbz_out = radar_dbz_out + Gaussian(dx_new / FWHMx, dy_new / FWHMy) * ConvertArray(
                        -ix, iy,
                        dbz)
                    radar_PIA_out = radar_PIA_out + Gaussian(dx_new / FWHMx, dy_new / FWHMy) * ConvertArray(
                        -ix, iy,
                        sj)
                    normalize = normalize + Gaussian(dx_new / FWHMx, dy_new / FWHMy)
            # ix is 0
            for iy in range(1, y_sub + 1):
                dx_new = 0
                # iy is positive
                dy_new = iy * dy
                radar_dbz_out = radar_dbz_out + Gaussian(dx_new / FWHMx, dy_new / FWHMy) * ConvertArray(0, iy,
                                                                                                        dbz)
                radar_PIA_out = radar_PIA_out + Gaussian(dx_new / FWHMx, dy_new / FWHMy) * ConvertArray(0, iy,
                                                                                                        sj)
                normalize = normalize + Gaussian(dx_new / FWHMx, dy_new / FWHMy)
                # iy is negative
                dy_new = -iy * dy
                radar_dbz_out = radar_dbz_out + Gaussian(dx_new / FWHMx, dy_new / FWHMy) * ConvertArray(0,
                                                                                                        -iy,
                                                                                                        dbz)
                radar_PIA_out = radar_PIA_out + Gaussian(dx_new / FWHMx, dy_new / FWHMy) * ConvertArray(0,
                                                                                                        -iy,
                                                                                                        sj)
                normalize = normalize + Gaussian(dx_new / FWHMx, dy_new / FWHMy)
            # iy is 0
            for ix in range(1, x_sub + 1):
                dy_new = 0
                # iy is positive
                dx_new = ix * dx
                radar_dbz_out = radar_dbz_out + Gaussian(dx_new / FWHMx, dy_new / FWHMy) * ConvertArray(ix, 0,
                                                                                                        dbz)
                radar_PIA_out = radar_PIA_out + Gaussian(dx_new / FWHMx, dy_new / FWHMy) * ConvertArray(ix, 0,
                                                                                                        sj)
                normalize = normalize + Gaussian(dx_new / FWHMx, dy_new / FWHMy)
                # ix is negative
                dx_new = -ix * dx
                radar_dbz_out = radar_dbz_out + Gaussian(dx_new / FWHMx, dy_new / FWHMy) * ConvertArray(-ix,
                                                                                                        0,
                                                                                                        dbz)
                radar_PIA_out = radar_PIA_out + Gaussian(dx_new / FWHMx, dy_new / FWHMy) * ConvertArray(-ix,
                                                                                                        0, sj)
                normalize = normalize + Gaussian(dx_new / FWHMx, dy_new / FWHMy)
            dbz_lyr = radar_dbz_out / normalize
            pia_lyr = radar_PIA_out / normalize
            # if dbz and pia > 0, dbz = 10 * log10(dbz), pia = -10 * log10(pia)
            # else dbz = pia = -np.inf
            # get index where dbz and pia <= 0
            dbz_lyr_negative_index = np.where(dbz_lyr <= 0)
            # pia_lyr_negative_index = np.where(pia_lyr <= 0)
            dbz_lyr[np.where(dbz_lyr > 0)] = 10 * np.log10(dbz_lyr[np.where(dbz_lyr > 0)])
            # pia_lyr[np.where(pia_lyr > 0)] = -10 * np.log10(pia_lyr[np.where(pia_lyr > 0)])
            # set dbz[index] and pia[index] to -np.inf
            dbz_lyr[dbz_lyr_negative_index] = -np.inf
            # pia_lyr[pia_lyr_negative_index] = -np.inf
            dbz_out.append(dbz_lyr)
            PIA_out.append(pia_lyr)
            process.advance(pid, advance=1)

    dbz_out = np.asarray(dbz_out).copy()
    PIA_out = np.asarray(PIA_out).copy()
    if save_data:
        np.savez('data/{}_coupling.npz'.format(filename), dbz=dbz_out, pia=PIA_out)
        print('{}_coupling.npz has been saved'.format(filename[:-3]))
    print('Finish!')

    return dbz_out, PIA_out
