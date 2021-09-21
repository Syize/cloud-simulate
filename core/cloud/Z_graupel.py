#!/usr/bin/python3


import numpy as np
from pandas import read_csv
from copy import deepcopy
from rich.progress import Progress
from ..tools import get_data


def Z_graupel(NCFile, RMax, Density, N0):
    # read file
    # Q Snow
    result = {'Z': [], 'K': []}
    # data shape is (50, 280, 280), change it if necessary
    start = 0
    end = 280
    QGraupel = read_csv('rcs_graupel.txt', sep='\\s+')
    QGraupel = np.asarray(QGraupel)[:, 1]
    QGraupel = QGraupel.reshape(1, -1)

    # K Graupel
    KGraupel = read_csv('kext_graupel.txt', sep='\\s+')
    KGraupel = np.asarray(KGraupel)[:, 1]
    KGraupel = KGraupel.reshape(1, -1)

    # nc
    data = NCFile
    with Progress() as process:
        pid = process.add_task("[red]Z_graupel is running...", total=280)
        for i in range(start, end):
            Graupel = get_data(data, 'QGRAUP', level=i)
            P = get_data(data, 'P', level=i)
            T = get_data(data, 'T', level=i) + 300
            PB = get_data(data, 'PB', level=i)
            T = T * np.power((P + PB) / 100000, 2 / 7)
            P1 = get_data(data, 'PB', level=i) / 100

            # parameters
            Dr = 5 / 100
            DD = Dr * 2
            Dgraupel = np.linspace(DD, RMax * 2, int((RMax * 2 - DD) / DD) + 1)

            # calculate
            # N
            Water = P1 * 100 / (287 * T)
            Water = Water * Graupel * 1000
            Water = np.ma.masked_values(Water, 0).copy()
            Lambda = np.power(np.pi * N0 * Density / (Water * 0.001), 0.25) * 0.001
            Lambda = Lambda.reshape(-1, 1).copy()
            N = N0 * np.exp(- Lambda * Dgraupel)

            # K
            K = KGraupel * N * DD
            K = np.sum(K, axis=1)
            result['K'].append(deepcopy(K.reshape(50, 280)))

            # Z
            Z = N * QGraupel * DD
            Z = np.sum(Z, axis=1)
            Z = Z.reshape(50, 280)
            Z = 0.319 ** 4 / np.pi ** 5 / 0.75 * Z
            result['Z'].append(deepcopy(np.asarray(Z).copy()))

            process.update(pid, advance=1)

    # stack array
    Z = np.stack(result['Z'], axis=1)
    K = np.stack(result['K'], axis=1)

    return Z, K
