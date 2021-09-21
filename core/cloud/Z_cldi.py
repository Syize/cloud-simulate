#!/usr/bin/python3


import numpy as np
from pandas import read_csv
from math import gamma
from copy import deepcopy
from rich.progress import Progress
from ..tools import get_data


def Z_cldi(NCFile, RMax, Density, N0, Mu):
    # read file
    # Q Snow
    result = {'Z': [], 'K': []}
    # data shape is (50, 280, 280), change it if necessary
    start = 0
    end = 280

    QCldi = read_csv('rcs_cldi.txt', sep='\\s+')
    QCldi = np.asarray(QCldi)[:, 1]
    QCldi = QCldi.reshape(1, -1)

    # K Cldi
    KCldi = read_csv('kext_cldi.txt', sep='\\s+')
    KCldi = np.asarray(KCldi)[:, 1]
    KCldi = KCldi.reshape(1, -1)

    # nc
    data = NCFile
    with Progress() as process:
        pid = process.add_task("[red]Z_cldi is running...", total=280)
        for i in range(start, end):
            Cldi = get_data(data, 'QICE', level=i)
            P = get_data(data, 'P', level=i)
            T = get_data(data, 'T', level=i) + 300
            PB = get_data(data, 'PB', level=i)
            T = T * np.power((P + PB) / 100000, 2 / 7)
            P1 = get_data(data, 'PB', level=i) / 100

            # parameters
            Dr = 5 / 1000
            DD = Dr * 2
            Drain = np.linspace(DD, RMax * 2, int((RMax * 2 - DD) / DD) + 1)

            # calculate
            # N
            Water = P1 * 100 / (287 * T)
            Water = Water * Cldi * 1000
            Nw = Water / 1000 / np.pi / Density * (3.67 / (N0 / 1000)) ** 4
            Nw = Nw.reshape(-1, 1).copy()
            N = Nw * 6 / 3.67 ** 4 * (3.67 + Mu) ** (Mu + 4) / gamma(Mu + 4) * (Drain / N0) ** Mu * np.exp(
                - (3.67 + Mu) * Drain / N0)

            # K
            K = KCldi * N * DD
            K = np.sum(K, axis=1)
            result['K'].append(deepcopy(K.reshape(50, 280)))

            # Z
            Z = N * QCldi * DD
            Z = np.sum(Z, axis=1)
            Z = Z.reshape(50, 280)
            Z = 0.319 ** 4 / np.pi ** 5 / 0.75 * Z
            result['Z'].append(deepcopy(np.asarray(Z).copy()))

            process.update(pid, advance=1)

    # stack array
    Z = np.stack(result['Z'], axis=1)
    K = np.stack(result['K'], axis=1)

    return Z, K
