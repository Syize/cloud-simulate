#!/usr/bin/python3

from copy import deepcopy
import numpy as np
from pandas import read_csv
from rich.progress import Progress
from ..tools import get_data


def Z_cldw(NCFile, RMax, Density, N0, Theta):
    # read file
    # Q Snow
    result = {'Z': [], 'K': []}
    # data shape is (50, 280, 280), change it if necessary
    start = 0
    end = 280

    QCldw = read_csv('rcs_cldw.txt', sep='\\s+')
    QCldw = np.asarray(QCldw)[:, 1]
    QCldw = QCldw.reshape(1, -1)

    # K Cldw
    KCldw = read_csv('kext_cldw.txt', sep='\\s+')
    KCldw = np.asarray(KCldw)[:, 1]
    KCldw = KCldw.reshape(1, -1)

    # nc
    data = NCFile
    with Progress() as process:
        pid = process.add_task("[red]Z_cldw is running...", total=280)
        for i in range(start, end):
            Cldw = get_data(data, 'QCLOUD', level=i)
            P = get_data(data, 'P', level=i)
            T = get_data(data, 'T', level=i) + 300
            PB = get_data(data, 'PB', level=i)
            T = T * np.power((P + PB) / 100000, 2 / 7)
            P1 = get_data(data, 'PB', level=i) / 100

            # parameters
            Dr = 5 / 10000
            DD = Dr * 2
            Drain = np.linspace(DD, RMax * 2, int((RMax * 2 - DD) / DD) + 1)

            # calculate
            # N
            Water = P1 * 100 / (287 * T)
            Water = Water * Cldw * 1000
            Water = Water.reshape(-1, 1).copy()
            N = 6 * Water / 1000 / (np.pi * np.sqrt(2 * np.pi) * Density * Theta * (N0 / 1000) ** 3) * np.exp(
                -4.5 * Theta ** 2) * np.exp(-0.5 * np.log(Drain / N0) ** 2 / Theta ** 2) / (Drain / 1000)

            # K
            K = KCldw * N * DD
            K = np.sum(K, axis=1)
            result['K'].append(deepcopy(K.reshape(50, 280)))

            # Z
            Z = N * QCldw * DD
            Z = np.sum(Z, axis=1)
            Z = Z.reshape(50, 280)
            Z = 0.319 ** 4 / np.pi ** 5 / 0.75 * Z
            result['Z'].append(deepcopy(np.asarray(Z).copy()))

            process.update(pid, advance=1)

    # stack array
    Z = np.stack(result['Z'], axis=1)
    K = np.stack(result['K'], axis=1)

    return Z, K
