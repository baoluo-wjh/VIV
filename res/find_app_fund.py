import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager
import numpy as np
from scipy.io import loadmat
from scipy.fftpack import fft, fftfreq, hilbert
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from collections.abc import Iterable
plt.rc('font', family='Times New Roman')
plt.rc('text', usetex = True)

def find_peak(arr, interval):
    ''' find local maxima '''
    peaks = []
    N = len(arr)
    mark = np.ones_like(arr, dtype='bool')
    for i in range(N):
        if not mark[i]:
            continue
        for j in range(1, interval//2):
            if i + j >= N:
                break
            if arr[i + j] < arr[i]:
                mark[i + j] = False
            else:
                mark[i] = False
                break
        if not mark[i]:
            continue
        for j in range(-interval // 2, 0):
            if i + j >= 0 and arr[i + j] > arr[i]:
                mark[i] = False
                break
        if mark[i]: 
            peaks.append(i)
    return peaks

def calc_derivative(arr, dt): 
    ''' dx/dt = (x[t+dt] - x[t-dt]) / (2*dt) '''
    a = np.zeros_like(arr)
    a[1 : -1] = (arr[2:] - arr[:-2]) / (2 * dt)
    a[0] = (arr[1] - arr[0]) / dt
    a[-1] = (arr[-1] - arr[-2]) / dt
    return a

def plot_time_spec_hil_deri(acc, j, jpg_name):

    fs = 50
    Ts = 1 / fs
    min_interval = 1
    min_stride = 1
    interval = int(min_interval * 60 / Ts)
    stride = int(min_stride * 60 / Ts)

    # fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 20))

    # ax_time = axes[0, 0]
    # ax_spec = axes[1, 0]
    # ax_hilb = axes[0, 1]
    # ax_deri = axes[1, 1]

    plt.figure(figsize=(15, 10))
    ax_time = plt.subplot2grid(shape=(2, 3), loc=(0, 0), colspan=2, rowspan=1)
    ax_spec = plt.subplot2grid(shape=(2, 3), loc=(1, 0), colspan=2, rowspan=1)
    ax_hilb = plt.subplot2grid(shape=(2, 3), loc=(0, 2), colspan=1, rowspan=1)
    ax_deri = plt.subplot2grid(shape=(2, 3), loc=(1, 2), colspan=1, rowspan=1)

    # time history
    a = acc[stride*j: interval + stride*j]
    # a -= a.mean()
    RMS = np.sqrt(np.sum(a ** 2) / interval)
    ax_time.plot(Ts * np.arange(interval), a, c="tab:green")
    ax_time.set_xlim(-1, min_interval*60 + 1)
    ax_time.tick_params(labelsize=27)
    # ax_time.set_xlabel("Time [s]", fontsize=27, color='k')
    ax_time.text(0.5, -0.15, "Time [s]", fontsize=27, color='k', 
                        transform=ax_time.transAxes, va="center", ha="center")
    # ax_time.set_ylabel("Acceleration [mg]", fontsize=27, color='k')
    ax_time.text(-0.1, 0.5, "Acceleration [mg]", fontsize=27, 
                        color='k', transform=ax_time.transAxes, va="center", 
                        ha="center", rotation=90)
    ax_time.text(0.5, 0.5, "RMS = %.0f" % RMS, fontsize=27, color='k', 
            transform=ax_time.transAxes, va="center", ha="center")

    # spectrum
    threshold = 0.4
    app_fund = 0.5
    spec = fft(a)
    f = fftfreq(interval, Ts)  # f.max() == fs/2 : Nesquite's sampling theory
    half_f = f[:(interval+1) // 2]
    half_spec = np.abs(spec[:(interval+1) // 2])
    half_spec[0] /= 2
    wave_filter = (half_f > threshold)
    half_spec *= wave_filter
    half_spec /= half_spec.max()
    ax_spec.plot(half_f, half_spec, c="tab:orange", alpha=0.5)
    ax_spec.set_ylim([-0.1, 1.1])
    ax_spec.set_yticks([0, 1])
    ax_spec.set_xlim(-1, 0.8 * fs/2 + 1)
    ax_spec.set_xticks(np.arange(1, 0.8 * fs/2 + 1, 2))
    ax_spec.tick_params(labelsize=27)
    ax_spec.set_xlabel("Frequency [Hz]", fontsize=27, color='k')
    # ax_spec.set_ylabel("Amplitude", fontsize=27, color='k')
    ax_spec.text(-0.1, 0.5, "Amplitude", fontsize=27, 
                    color='k', transform=ax_spec.transAxes, va="center", 
                    ha="center", rotation=90)
    # calc PRS
    FSM_interval = int(1.5 * app_fund / half_f[1])
    peaks = np.array(find_peak(half_spec, FSM_interval))
    sorted_peaks = sorted(peaks, key=lambda i: half_spec[i])
    F1 = half_f[sorted_peaks[-1]]
    F2 = half_f[sorted_peaks[-2]]
    A1 = half_spec[sorted_peaks[-1]]
    A2 = half_spec[sorted_peaks[-2]]
    PRS = A2 / A1
    diff_freq = np.abs(F1 - F2)
    # ax_spec.text(0.7, 0.7, "PRS=\n%.4f" % PRS, fontsize=27, color='k', 
    #         transform=ax_spec.transAxes, va="center", ha="center")
    ax_spec.text(0.7, 0.7, "diff_freq=\n%.4f" % diff_freq, fontsize=27, color='k', 
            transform=ax_spec.transAxes, va="center", ha="center")
    ax_spec.scatter([F1, F2], [A1, A2], c="tab:purple", s=128, marker='*')

    # H.T.
    HT = hilbert(a)
    x = a / np.abs(a).max()
    y = HT / np.abs(HT).max()
    CR = np.mean(x)
    CI = np.mean(y)
    Rh = np.sqrt((x-CR) ** 2 + (y-CI) ** 2)
    HCH = Rh[interval//1000: interval-interval//1000].min() / \
            Rh[interval//1000: interval-interval//1000].max()
    ax_hilb.axis("equal")
    ax_hilb.scatter(x[interval//1000: interval-interval//1000], 
            y[interval//1000: interval-interval//1000], c='tab:pink')
    ax_hilb.text(0.5, 0.5, "HCH=\n%.4f" % HCH, fontsize=27, color='k', 
            transform=ax_hilb.transAxes, va="center", ha="center")
    MARGIN = 1.05
    ax_hilb.set_xlim([-MARGIN, MARGIN])
    ax_hilb.set_ylim([-MARGIN, MARGIN])
    ax_hilb.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax_hilb.set_yticks([-1, 1])
    ax_hilb.tick_params(labelsize=27)
    # ax_hilb.set_xlabel("Origin", fontsize=27, color='k')
    ax_hilb.text(0.5, -0.15, "Origin", fontsize=27, color='k', 
                        transform=ax_hilb.transAxes, va="center", ha="center")
    # ax_hilb.set_ylabel("Hilbert", fontsize=27, color='k')
    ax_hilb.text(-0.1, 0.5, "Hilbert", fontsize=27, 
                color='k', transform=ax_hilb.transAxes, va="center", 
                ha="center", rotation=90)

    # derivative
    dt = Ts
    x = a / np.abs(a).max()
    y = calc_derivative(x, dt)
    y /= np.abs(y).max()
    DCR = np.mean(x)
    DCI = np.mean(y)
    Ry = np.sqrt((x - DCR) ** 2 + (y - DCI) ** 2)
    HCD = Ry[interval//1000: interval-interval//1000].min() / \
            Ry[interval//1000: interval-interval//1000].max()
    ax_deri.axis("equal")
    ax_deri.scatter(x[interval//1000: interval-interval//1000], 
            y[interval//1000: interval-interval//1000], c='tab:olive')
    ax_deri.text(0.5, 0.5, "HCD=\n%.4f" % HCD, fontsize=27, color='k', 
            transform=ax_deri.transAxes, va="center", ha="center")
    ax_deri.set_xlim([-MARGIN, MARGIN])
    ax_deri.set_ylim([-MARGIN, MARGIN])
    ax_deri.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax_deri.set_yticks([-1, 1])
    ax_deri.tick_params(labelsize=27)
    ax_deri.set_xlabel("Origin", fontsize=27, color='k')
    # ax_deri.set_ylabel("Derivative", fontsize=27, color='k')
    ax_deri.text(-0.1, 0.5, "Derivative", fontsize=27, 
                color='k', transform=ax_deri.transAxes, va="center", 
                ha="center", rotation=90)

    plt.savefig(jpg_name, bbox_inches="tight", pad_inches=0.1)
    plt.close()

def main():
    mat_file = "/media/ps/wjh/铜陵桥数据22.7-23.2/2022-07-01/2022-07-01 00-VIC.mat"
    acc = loadmat(mat_file)["data"]
    cable_num = acc.shape[1]
    for i in range(cable_num):
        if i != 34:
            continue
        a = acc[:, i]
        plot_time_spec_hil_deri(a, 1, "./%02d.jpg" % (i+1))

main()
