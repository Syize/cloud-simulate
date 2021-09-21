#!/usr/bin/python3


import numpy as np
from pandas import read_csv
from copy import deepcopy
from rich.progress import Progress
from ..tools import get_data


def Z_rain(NCFile, RMax, Density, N0):
    # read file
    # Q Snow
    result = {'Z': [], 'K': []}
    # data shape is (50, 280, 280), change it if necessary
    start = 0
    end = 280

    QRain = read_csv('rcs_rain.txt', sep='\\s+')
    QRain = np.asarray(QRain)[:, 1]
    QRain = QRain.reshape(1, -1)

    # K Rain
    KRain = read_csv('kext_rain.txt', sep='\\s+')
    KRain = np.asarray(KRain)[:, 1]
    KRain = KRain.reshape(1, -1)

    # nc
    data = NCFile

    with Progress() as process:
        pid = process.add_task("[red]Z_rain is running...", total=280)
        for i in range(start, end):
            Rain = get_data(data, 'QRAIN', level=i)
            P = get_data(data, 'P', level=i)
            T = get_data(data, 'T', level=i) + 300
            PB = get_data(data, 'PB', level=i)
            T = T * np.power((P + PB) / 100000, 2 / 7)
            P1 = get_data(data, 'PB', level=i) / 100

            # parameters
            Dr = 5 / 100
            DD = Dr * 2
            Drain = np.linspace(DD, RMax * 2, int((RMax * 2 - DD) / DD) + 1)

            # calculate
            # N
            Water = P1 * 100 / (287 * T)
            Water = Water * Rain * 1000
            Water = np.ma.masked_values(Water, 0).copy()
            Lambda = np.power(np.pi * N0 * Density / (Water * 0.001), 0.25) * 0.001
            Lambda = Lambda.reshape(-1, 1).copy()
            N = N0 * np.exp(- Lambda * Drain)

            # K
            K = KRain * N * DD
            K = np.sum(K, axis=1)
            result['K'].append(deepcopy(K.reshape(50, 280)))

            # Z
            Z = N * QRain * DD
            Z = np.sum(Z, axis=1)
            Z = Z.reshape(50, 280)
            Z = 0.319 ** 4 / np.pi ** 5 / 0.75 * Z
            result['Z'].append(deepcopy(np.asarray(Z).copy()))

            process.update(pid, advance=1)

    # stack array
    Z = np.stack(result['Z'], axis=1)
    K = np.stack(result['K'], axis=1)
    # save to .npz
    np.savez('data/cldi.npz', Z=Z, K=K)

    return Z, K
