# -*- coding:utf-8 -*-

from netCDF4 import Dataset
import numpy as np
from rich.progress import Progress
from math import exp, log
import matplotlib.pyplot as plt
from matplotlib import ticker
from os import chdir, getcwd, mkdir


def get_data(nc, name, level=None):
    """
    This function is used to get data in nc file, along the axis 1
    If you want to stack arrays back to 3D, use np.stack(list: [array1, array2 ...], axis=1)
    For example
    a = [[1,2],     b = [[5,6],
         [3,4]]          [7,8]]
    np.stack([a,b], axis=1) = [[[1, 2],
                                [5, 6]],

                               [[3, 4],
                                [7, 8]]]
    :param nc:
    :param name:
    :param level:
    :return:
    """
    if level is not None:
        return np.asarray(nc[name])[0][:, level, :]
    else:
        raise Exception('You must give level to get specific layer of data')


def get_data_axis(nc, name, level=None, axis=None):
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


def read_data(filename: str, data_type: str):
    """
    Read data from npz file, filename is nc file name, which ends with .nc, data_type include: Z, K, SJ, coupling
    :param filename: nc file name
    :param data_type: Z, K, SJ or coupling
    :return: dict include data
    """
    return np.load('data/{0}_{1}.npz'.format(filename[:-3], data_type))


def _get_Z_data(data, level, axis):
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


def Attenuation(K, H):
    """
    This function is used to simplify calculate process although in fact it doesn't.
    :param K:
    :param H:
    :return:
    """
    return 20 * 0.4343 * K * H


def Gaussian(X, Y):
    """
    2D Gaussian convolution kernel
    :param X:
    :param Y:
    :return:
    """
    return exp(-(X ** 2 + Y ** 2) * 4 * log(2))


def ConvertArray(x_toward, y_toward, old_array):
    """
    对矩阵做平移变换，边缘补0
    :param x_toward: 横向平移量
    :param y_toward: 纵向平移量
    :param old_array: 原矩阵
    :return: 平移后矩阵
    """
    array_x, array_y = old_array.shape
    new_array = old_array
    if x_toward > 0:
        new_array = np.hstack((np.zeros((array_x, x_toward)), new_array[:, :-x_toward]))
    elif x_toward == 0:
        pass
    else:
        x_toward = -x_toward
        new_array = np.hstack((new_array[:, x_toward:], np.zeros((array_x, x_toward))))
    if y_toward > 0:
        new_array = np.vstack((new_array[y_toward:, :], np.zeros((y_toward, array_y))))
    elif y_toward == 0:
        pass
    else:
        y_toward = -y_toward
        new_array = np.vstack((np.zeros((y_toward, array_y)), new_array[:-y_toward, :]))
    return new_array


def Calculate_SJ(filename, k_total):
    """
    计算衰减系数
    :param filename: nc file name
    :param k_total:
    :return: 整个 3D 数据的衰减系数
    """
    data = Dataset(filename)
    # Get PHB and PH data to calculate height to calculate attenuation
    PHB = np.asarray(data['PHB'])[0]
    PH = np.asarray(data['PH'])[0]
    Height = (PH + PHB) / 9.81
    Height = Height[:50, :280, :280]
    z_num, x_num, y_num = Height.shape

    # concatenate can stack two array of any dimension in any axis
    Height2 = np.concatenate((np.asarray([np.zeros((280, 280))]), Height[:-1, :, :]))
    # H_phase is Δh between two levels
    H_phase = (Height - Height2) / 1000

    # attenuation
    # Use sum to calculate attenuation
    with Progress() as progress:
        pid = progress.add_task('[red]Calculating attenuation...', total=x_num)
        SJ = []
        for _num in range(x_num):
            k_total_lyr = k_total[:, _num, :]
            h_phase_lyr = H_phase[:, _num, :]
            _SJ = [np.sum(Attenuation(k_total_lyr, h_phase_lyr), axis=0)]
            for i in range(1, h_phase_lyr.shape[0]):
                _SJ = np.concatenate(
                    (_SJ, np.asarray([np.sum(Attenuation(k_total_lyr[i:, :], h_phase_lyr[i:, :]), axis=0)])))
            SJ.append(_SJ.reshape(z_num, 1, y_num))
            progress.advance(pid, advance=1)
    SJ = np.concatenate(SJ, axis=1)

    return SJ


def Calculate_SJ_test(filename, K: dict, mode=0, save_data=0):
    """
    计算衰减系数
    :param mode: If data file exists, function will not calculate it again. Set this to 1 to force function re-calculate
    :param save_data: If you want to save data in file, set this to 1
    :param filename: nc file name
    :param K:
    :return: 整个 3D 数据的衰减系数
    """
    # check if data exists
    if not mode:
        try:
            f = open('data/{0}_SJ.npz'.format(filename[:-3]), 'r+b')
            f.close()
            print('Attenuation data: {}_SJ.npz exists, so pass it\nif you want to re-calculate it, set mode=1\n'.format(
                filename[:-3]))
            return np.load('data/{0}_SJ.npz'.format(filename[:-3]))
        except FileNotFoundError:
            pass

    data = Dataset(filename)
    # Get data to calculate height to calculate attenuation
    PHB = np.asarray(data['PHB'])[0]
    PH = np.asarray(data['PH'])[0]
    Height = (PH + PHB) / 9.81
    Height = Height[:50, :280, :280]
    height_z, height_x, height_y = Height.shape
    k_total = K['total']
    k_cldi = K['cldi']
    k_cldw = K['cldw']
    k_rain = K['rain']
    k_snow = K['snow']
    k_graupel = K['graupel']

    # concatenate can stack two array of any dimension in any axis
    Height2 = np.concatenate((np.asarray([np.zeros((280, 280))]), Height[:-1, :, :]))
    # H_phase is Δh between two levels
    H_phase = (Height - Height2) / 1000

    # attenuation
    # Use sum to calculate attenuation
    with Progress() as progress:
        pid = progress.add_task('[red]Calculating attenuation...', total=height_x)
        total_sj = []
        cldi_sj = []
        cldw_sj = []
        rain_sj = []
        snow_sj = []
        graupel_sj = []
        for _num in range(height_x):  # 每一层数据
            # Get No._num layer of data
            total_lyr = k_total[:, _num, :]
            cldi_lyr = k_cldi[:, _num, :]
            cldw_lyr = k_cldw[:, _num, :]
            rain_lyr = k_rain[:, _num, :]
            snow_lyr = k_snow[:, _num, :]
            graupel_lyr = k_graupel[:, _num, :]
            h_phase_lyr = H_phase[:, _num, :]

            # calculate the sj on top line
            _total_sj = [np.sum(Attenuation(total_lyr, h_phase_lyr), axis=0)]
            _cldi_sj = [np.sum(Attenuation(cldi_lyr, h_phase_lyr) / 10 ** 9, axis=0)]
            _cldw_sj = [np.sum(Attenuation(cldw_lyr, h_phase_lyr) / 10 ** 9, axis=0)]
            _rain_sj = [np.sum(Attenuation(rain_lyr, h_phase_lyr) / 10 ** 9, axis=0)]
            _snow_sj = [np.sum(Attenuation(snow_lyr, h_phase_lyr) / 10 ** 9, axis=0)]
            _graupel_sj = [np.sum(Attenuation(graupel_lyr, h_phase_lyr) / 10 ** 9, axis=0)]

            for i in range(1, h_phase_lyr.shape[0]):  # 循环计算衰减
                _total_sj = np.concatenate(
                    (
                        _total_sj,
                        np.asarray([np.sum(Attenuation(total_lyr[i:, :], h_phase_lyr[i:, :]), axis=0)])))
                _cldi_sj = np.concatenate(
                    (
                        _cldi_sj,
                        np.asarray([np.sum(Attenuation(cldi_lyr[i:, :], h_phase_lyr[i:, :]) / 10 ** 9, axis=0)])))
                _cldw_sj = np.concatenate(
                    (
                        _cldw_sj,
                        np.asarray([np.sum(Attenuation(cldw_lyr[i:, :], h_phase_lyr[i:, :]) / 10 ** 9, axis=0)])))
                _rain_sj = np.concatenate(
                    (
                        _rain_sj,
                        np.asarray([np.sum(Attenuation(rain_lyr[i:, :], h_phase_lyr[i:, :]) / 10 ** 9, axis=0)])))
                _snow_sj = np.concatenate(
                    (
                        _snow_sj,
                        np.asarray([np.sum(Attenuation(snow_lyr[i:, :], h_phase_lyr[i:, :]) / 10 ** 9, axis=0)])))
                _graupel_sj = np.concatenate(
                    (_graupel_sj,
                     np.asarray([np.sum(Attenuation(graupel_lyr[i:, :], h_phase_lyr[i:, :]) / 10 ** 9, axis=0)])))

            total_sj.append(_total_sj.reshape(height_z, 1, height_y))
            cldi_sj.append(_cldi_sj.reshape(height_z, 1, height_y))
            cldw_sj.append(_cldw_sj.reshape(height_z, 1, height_y))
            rain_sj.append(_rain_sj.reshape(height_z, 1, height_y))
            snow_sj.append(_snow_sj.reshape(height_z, 1, height_y))
            graupel_sj.append(_graupel_sj.reshape(height_z, 1, height_y))

            progress.advance(pid, advance=1)

    total_sj = np.concatenate(total_sj, axis=1)
    cldi_sj = np.concatenate(cldi_sj, axis=1)
    cldw_sj = np.concatenate(cldw_sj, axis=1)
    rain_sj = np.concatenate(rain_sj, axis=1)
    snow_sj = np.concatenate(snow_sj, axis=1)
    graupel_sj = np.concatenate(graupel_sj, axis=1)

    if save_data:
        np.savez('data/{}_SJ.npz'.format(filename[:-3]), total=total_sj, cldi=cldi_sj, cldw=cldw_sj, rain=rain_sj,
                 snow=snow_sj, graupel=graupel_sj)
        print('All {0} attenuation data has been saved in data folder'.format(filename))
    print('Finish!')
    return {
        'total': total_sj,
        'cldi': cldi_sj,
        'cldw': cldw_sj,
        'rain': rain_sj,
        'snow': snow_sj,
        'graupel': graupel_sj
    }


def draw_radar(filename, dbz, pia, axis, level):
    """
    draw picture after radar coupling finish
    :param filename: nc file name
    :param dbz: radar coupling dbz data
    :param pia: radar coupling attenuation data
    :param axis: which direction you cut data
    :param level: which layer of data along the axis
    :return: None
    """
    dataset = Dataset(filename)
    if axis == 1:
        X = np.asarray(dataset['XLONG'])[0][level, :]
        X_label = 'Longitude'
    else:
        X = np.asarray(dataset['XLAT'])[0][:, level]
        X_label = 'Latitude'

    PHB = get_data_axis(dataset, 'PHB', level=level, axis=axis)
    PH = get_data_axis(dataset, 'PH', level=level, axis=axis)
    dbz = dbz[:, :, level]
    pia = pia[:, :, level]
    Height = (PH + PHB) / 9.81
    Height = Height[:50, :280]

    X = (np.zeros(Height.shape) + 1) * X

    plt.figure()
    dbz[dbz < -35] = -35
    dbz[dbz > 30] = 30
    plt.contourf(X, Height / 1000, dbz, levels=np.linspace(-35, 30, 65), cmap='jet')
    plt.title('Radar Coupling dbz before attenuation')
    cbr = plt.colorbar()
    cbr.locator = ticker.MaxNLocator(nbins=12)
    cbr.set_ticks(np.arange(-30, 30, 5))
    plt.xlabel(X_label)
    plt.ylabel('Height (km)')
    plt.savefig('res/radar_dbz_before.png')

    plt.figure()
    plt.contourf(X, Height / 1000, pia, cmap='jet')
    plt.title('Radar Coupling pia')
    plt.colorbar()
    plt.xlabel(X_label)
    plt.ylabel('Height (km)')
    plt.savefig('res/radar_pia.png')

    plt.figure()
    dbz_pia = dbz - pia
    dbz_pia[dbz_pia < -35] = -35
    dbz_pia[dbz_pia > 30] = 30
    plt.contourf(X, Height / 1000, dbz_pia, levels=np.linspace(-35, 30, 65), cmap='jet')
    plt.title('Radar Coupling dbz after attenuation')
    cbr = plt.colorbar()
    cbr.locator = ticker.MaxNLocator(nbins=12)
    cbr.set_ticks(np.arange(-30, 30, 5))
    plt.xlabel(X_label)
    plt.ylabel('Height (km)')
    plt.savefig('res/radar_dbz_after.png')


def draw_cloud(filename, Z: dict, K: dict, axis, level, mode=0, save_data=0):
    """
    use data draw picture
    :param save_data:
    :param mode:
    :param level: dara layer
    :param axis: X axis (0) or Y axis (1)
    :param filename: nc filename
    :param Z: Z data
    :param K: K data
    :return: None
    """
    # get data and draw picture
    # Get longitude and latitude to plot pictures
    Data = Dataset(filename)
    if axis == 1:
        X = np.asarray(Data['XLONG'])[0][level, :]
        X_Label = 'Longitude'
    else:
        X = np.asarray(Data['XLAT'])[0][:, level]
        X_Label = 'Latitude'

    # Get data
    ZTotal = Z['total']
    ZCldi = Z['cldi']
    ZCldw = Z['cldw']
    ZRain = Z['rain']
    ZSnow = Z['snow']
    ZGraupel = Z['graupel']

    # Get PHB and PH data to calculate height to calculate attenuation
    PHB = get_data_axis(Data, 'PHB', level=level, axis=axis)
    PH = get_data_axis(Data, 'PH', level=level, axis=axis)
    Height = (PH + PHB) / 9.81
    Height = Height[:ZRain.shape[0], :ZRain.shape[1]]

    # get specific data and convert data to log10() * 10
    ZCldi = _get_Z_data(ZCldi, level, axis)
    ZCldw = _get_Z_data(ZCldw, level, axis)
    ZSnow = _get_Z_data(ZSnow, level, axis)
    ZRain = _get_Z_data(ZRain, level, axis)
    ZGraupel = _get_Z_data(ZGraupel, level, axis)
    ZTotal = _get_Z_data(ZTotal, level, axis)

    ZCldw = np.log10(ZCldw) * 10
    ZCldi = np.log10(ZCldi) * 10
    ZSnow = np.log10(ZSnow) * 10
    ZRain = np.log10(ZRain) * 10
    ZGraupel = np.log10(ZGraupel) * 10

    sj = Calculate_SJ_test(filename, K, mode=mode, save_data=save_data)
    SJ = sj['total']
    CldwSJ = sj['cldw']
    CldiSJ = sj['cldi']
    RainSJ = sj['rain']
    SnowSJ = sj['snow']
    GraupelSJ = sj['graupel']

    SJ = _get_Z_data(SJ, level, axis)
    CldwSJ = _get_Z_data(CldwSJ, level, axis)
    CldiSJ = _get_Z_data(CldiSJ, level, axis)
    RainSJ = _get_Z_data(RainSJ, level, axis)
    SnowSJ = _get_Z_data(SnowSJ, level, axis)
    GraupelSJ = _get_Z_data(GraupelSJ, level, axis)
    # ###########################################################################
    # plot
    # The code below is to draw pictures
    work_path = getcwd()
    try:
        chdir('res')
        chdir(work_path)
    except FileNotFoundError:
        mkdir('res')
    # generate X coordinates
    X = (np.zeros(Height.shape) + 1) * X

    # Total
    plt.figure()
    # If a data is out of range [-35, 30], set it -35 or 35 to correct the color bar
    ZTotal[ZTotal < -35] = -35
    ZTotal[ZTotal > 30] = 30
    plt.contourf(X, Height / 1000, ZTotal, levels=np.linspace(-35, 30, 65), cmap='jet')
    plt.title('ZTotal before attenuation')
    # Set color bar range
    cbr = plt.colorbar()
    cbr.locator = ticker.MaxNLocator(nbins=12)
    cbr.set_ticks(np.arange(-30, 30, 5))
    plt.xlabel(X_Label)
    plt.ylabel('Height (km)')
    plt.savefig('res/Total_before.png')

    plt.figure()
    plt.contourf(X, Height / 1000, SJ, cmap='jet')
    plt.title('ZTotal attenuation')
    plt.colorbar()
    plt.xlabel(X_Label)
    plt.ylabel('Height (km)')
    plt.savefig('res/Total_attenuation.png')

    plt.figure()
    ZTotal_SJ = ZTotal - SJ
    ZTotal_SJ[ZTotal_SJ < -35] = -35
    ZTotal_SJ[ZTotal_SJ > 30] = 30
    plt.contourf(X, Height / 1000, ZTotal_SJ, levels=np.linspace(-35, 30, 65), cmap='jet')
    plt.title('ZTotal after attenuation')
    cbr = plt.colorbar()
    cbr.locator = ticker.MaxNLocator(nbins=12)
    cbr.set_ticks(np.arange(-30, 30, 5))
    plt.xlabel(X_Label)
    plt.ylabel('Height (km)')
    plt.savefig('res/Total_after.png')

    # Cldw
    plt.figure()
    ZCldw[ZCldw < -50] = -50
    ZCldw[ZCldw > -20] = -20
    plt.contourf(X, Height / 1000, ZCldw, levels=np.linspace(-50, -20, 30), cmap='jet')
    plt.title('ZCldw before attenuation')
    cbr = plt.colorbar()
    cbr.locator = ticker.MaxNLocator(nbins=5)
    cbr.set_ticks(np.arange(-50, -20, 5))
    plt.xlabel(X_Label)
    plt.ylabel('Height (km)')
    plt.savefig('res/Cldw_before.png')

    plt.figure()
    plt.contourf(X, Height / 1000, CldwSJ, cmap='jet')
    plt.title('ZCldw attenuation')
    plt.colorbar()
    plt.xlabel(X_Label)
    plt.ylabel('Height (km)')
    plt.savefig('res/Cldw_attenuation.png')

    plt.figure()
    ZCldw_SJ = ZCldw - CldwSJ
    ZCldw_SJ[ZCldw_SJ < -50] = -50
    ZCldw_SJ[ZCldw_SJ > -20] = -20
    plt.contourf(X, Height / 1000, ZCldw_SJ, levels=np.linspace(-50, -20, 30), cmap='jet')
    plt.title('ZCldw after attenuation')
    cbr = plt.colorbar()
    cbr.locator = ticker.MaxNLocator(nbins=5)
    cbr.set_ticks(np.arange(-50, -20, 5))
    plt.xlabel(X_Label)
    plt.ylabel('Height (km)')
    plt.savefig('res/Cldw_after.png')

    # Cldi
    plt.figure()
    ZCldi[ZCldi < -50] = -50
    ZCldi[ZCldi > -10] = -10
    plt.contourf(X, Height / 1000, ZCldi, levels=np.linspace(-50, -10, 40), cmap='jet')
    plt.title('ZCldi before attenuation')
    cbr = plt.colorbar()
    cbr.locator = ticker.MaxNLocator(nbins=7)
    cbr.set_ticks(np.arange(-50, -10, 5))
    plt.xlabel(X_Label)
    plt.ylabel('Height (km)')
    plt.savefig('res/Cldi_before.png')

    plt.figure()
    plt.contourf(X, Height / 1000, CldiSJ, cmap='jet')
    plt.title('ZCldi attenuation')
    plt.colorbar()
    plt.xlabel(X_Label)
    plt.ylabel('Height (km)')
    plt.savefig('res/Cldi_attenuation.png')

    plt.figure()
    ZCldi_SJ = ZCldi - CldiSJ
    ZCldi_SJ[ZCldi_SJ < -50] = -50
    ZCldi_SJ[ZCldi_SJ > -10] = -10
    plt.contourf(X, Height / 1000, ZCldi_SJ, levels=np.linspace(-50, -10, 40), cmap='jet')
    plt.title('ZCldi after attenuation')
    cbr = plt.colorbar()
    cbr.locator = ticker.MaxNLocator(nbins=7)
    cbr.set_ticks(np.arange(-50, -10, 5))
    plt.xlabel(X_Label)
    plt.ylabel('Height (km)')
    plt.savefig('res/Cldi_after.png')

    # Rain
    plt.figure()
    ZRain[ZRain < -35] = -35
    ZRain[ZRain > 30] = 30
    plt.contourf(X, Height / 1000, ZRain, levels=np.linspace(-35, 30, 65), cmap='jet')
    plt.title('ZRain before attenuation')
    cbr = plt.colorbar()
    cbr.locator = ticker.MaxNLocator(nbins=12)
    cbr.set_ticks(np.arange(-30, 30, 5))
    plt.xlabel(X_Label)
    plt.ylabel('Height (km)')
    plt.savefig('res/Rain_before.png')

    plt.figure()
    plt.contourf(X, Height / 1000, RainSJ, cmap='jet')
    plt.title('ZRain attenuation')
    plt.colorbar()
    plt.xlabel(X_Label)
    plt.ylabel('Height (km)')
    plt.savefig('res/Rain_attenuation.png')

    plt.figure()
    ZRain_SJ = ZRain - RainSJ
    ZRain_SJ[ZRain_SJ < -35] = -35
    ZRain_SJ[ZRain_SJ > 30] = 30
    plt.contourf(X, Height / 1000, ZRain_SJ, levels=np.linspace(-35, 30, 65), cmap='jet')
    plt.title('ZRain after attenuation')
    cbr = plt.colorbar()
    cbr.locator = ticker.MaxNLocator(nbins=12)
    cbr.set_ticks(np.arange(-30, 30, 5))
    plt.xlabel(X_Label)
    plt.ylabel('Height (km)')
    plt.savefig('res/Rain_after.png')

    # Snow
    plt.figure()
    ZSnow[ZSnow < -35] = -35
    ZSnow[ZSnow > 30] = 30
    plt.contourf(X, Height / 1000, ZSnow, levels=np.linspace(-35, 30, 65), cmap='jet')
    plt.title('ZSnow before attenuation')
    cbr = plt.colorbar()
    cbr.locator = ticker.MaxNLocator(nbins=12)
    cbr.set_ticks(np.arange(-30, 30, 5))
    plt.xlabel(X_Label)
    plt.ylabel('Height (km)')
    plt.savefig('res/Snow_before.png')

    plt.figure()
    plt.contourf(X, Height / 1000, SnowSJ, cmap='jet')
    plt.title('ZSnow attenuation')
    plt.colorbar()
    plt.xlabel(X_Label)
    plt.ylabel('Height (km)')
    plt.savefig('res/Snow_attenuation.png')

    plt.figure()
    ZSnow_SJ = ZSnow - SnowSJ
    ZSnow_SJ[ZSnow_SJ < -35] = -35
    ZSnow_SJ[ZSnow_SJ > 30] = 30
    plt.contourf(X, Height / 1000, ZSnow_SJ, levels=np.linspace(-35, 30, 65), cmap='jet')
    plt.title('ZSnow after attenuation')
    cbr = plt.colorbar()
    cbr.locator = ticker.MaxNLocator(nbins=12)
    cbr.set_ticks(np.arange(-30, 30, 5))
    plt.xlabel(X_Label)
    plt.ylabel('Height (km)')
    plt.savefig('res/Snow_after.png')

    # Graupel
    plt.figure()
    ZGraupel[ZGraupel < -35] = -35
    ZGraupel[ZGraupel > 30] = 30
    plt.contourf(X, Height / 1000, ZGraupel, levels=np.linspace(-35, 30, 65), cmap='jet')
    plt.title('ZGraupel before attenuation')
    cbr = plt.colorbar()
    cbr.locator = ticker.MaxNLocator(nbins=12)
    cbr.set_ticks(np.arange(-30, 30, 5))
    plt.xlabel(X_Label)
    plt.ylabel('Height (km)')
    plt.savefig('res/Graupel_before.png')

    plt.figure()
    plt.contourf(X, Height / 1000, GraupelSJ, cmap='jet')
    plt.title('ZGraupel attenuation')
    plt.colorbar()
    plt.xlabel(X_Label)
    plt.ylabel('Height (km)')
    plt.savefig('res/Graupel_attenuation.png')

    plt.figure()
    ZGraupel_SJ = ZGraupel - GraupelSJ
    ZGraupel_SJ[ZGraupel_SJ < -35] = -35
    ZGraupel_SJ[ZGraupel_SJ > 30] = 30
    plt.contourf(X, Height / 1000, ZGraupel_SJ, levels=np.linspace(-35, 30, 65), cmap='jet')
    plt.title('ZGraupel after attenuation')
    cbr = plt.colorbar()
    cbr.locator = ticker.MaxNLocator(nbins=12)
    cbr.set_ticks(np.arange(-30, 30, 5))
    plt.xlabel(X_Label)
    plt.ylabel('Height (km)')
    plt.savefig('res/Graupel_after.png')

    # plt.show()

    print('Finish! All picture has been saved in "res" folder')

    return sj


def __dir__():
    """
    Overwrite __dir__ function, only show functions below
    :return:
    """
    return ['get_data', 'Attenuation', 'Gaussian', 'ConvertArray', 'Calculate_SJ', 'get_data_axis', 'draw_radar',
            'draw_cloud']


# define API, you can use from tools import * to import those functions
__all__ = ['get_data', 'Attenuation', 'Gaussian', 'ConvertArray', 'Calculate_SJ', 'Calculate_SJ_test', 'read_data',
           'get_data_axis', 'draw_radar', 'draw_cloud']
