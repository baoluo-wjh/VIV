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

def calc_derivative(arr, dt): 
    ''' dx/dt = (x[t+dt] - x[t-dt]) / (2*dt) '''
    a = np.zeros_like(arr)
    a[1 : -1] = (arr[2:] - arr[:-2]) / (2 * dt)
    a[0] = (arr[1] - arr[0]) / dt
    a[-1] = (arr[-1] - arr[-2]) / dt
    return a

def plot_hilbert_ideal():
    N = 50 * 60 * 3  # 10000
    t = np.arange(N) * 0.02

    # # step
    # x1 = np.zeros(N)
    # x1[7500:] = 10000
    # # x1 -= x1.mean()
    # y1 = hilbert(x1)
    # x1 /= np.max(np.abs(x1))
    # y1 /= np.max(np.abs(y1))
    # plt.figure(figsize=(20, 10))
    # ax1 = plt.subplot2grid(shape=(2, 4), loc=(0, 0), colspan=2, rowspan=1)
    # ax2 = plt.subplot2grid(shape=(2, 4), loc=(1, 0), colspan=2, rowspan=1)
    # ax3 = plt.subplot2grid(shape=(2, 4), loc=(0, 2), colspan=2, rowspan=2)
    # ax1.set_xlabel("t [s]", size=28)
    # ax1.set_ylabel("Acceleration [Normalized]", size=28)
    # ax1.tick_params(labelsize=24)
    # ax2.set_xlabel("t [s]", size=28)
    # ax2.set_ylabel("Hilbert [Normalized]", size=28)
    # ax2.tick_params(labelsize=24)
    # ax3.tick_params(labelsize=24)
    # ax1.plot(t, x1, c="tab:blue")
    # ax2.plot(t, y1, c="tab:blue")
    # ax3.scatter(x1[N//1000: N-N//1000], y1[N//1000: N-N//1000], s=1, marker='*')
    # ax3.set_xlim([-1.05, 1.05])
    # ax3.set_ylim([-1.05, 1.05])
    # plt.tight_layout()
    # plt.savefig("./hilbert_ideal_step.jpg")
    # plt.close()

    # # dirac
    # x2 = np.zeros(10000)
    # x2[7500: 7510] = 10000
    # # x2 -= x2.mean()
    # y2= hilbert(x2)
    # x2 /= np.max(np.abs(x2))
    # y2 /= np.max(np.abs(y2))
    # plt.figure(figsize=(20, 10))
    # ax1 = plt.subplot2grid(shape=(2, 4), loc=(0, 0), colspan=2, rowspan=1)
    # ax2 = plt.subplot2grid(shape=(2, 4), loc=(1, 0), colspan=2, rowspan=1)
    # ax3 = plt.subplot2grid(shape=(2, 4), loc=(0, 2), colspan=2, rowspan=2)
    # ax1.set_xlabel("t [s]", size=28)
    # ax1.set_ylabel("Acceleration [Normalized]", size=28)
    # ax1.tick_params(labelsize=24)
    # ax2.set_xlabel("t [s]", size=28)
    # ax2.set_ylabel("Hilbert [Normalized]", size=28)
    # ax2.tick_params(labelsize=24)
    # ax3.tick_params(labelsize=24)
    # ax1.plot(t, x2, c="tab:blue")
    # ax2.plot(t, y2, c="tab:blue")
    # ax3.scatter(x1[N//1000: N-N//1000], y1[N//1000: N-N//1000], s=1, marker='*')
    # ax3.set_xlim([-1.05, 1.05])
    # ax3.set_ylim([-1.05, 1.05])
    # plt.tight_layout()
    # plt.savefig("./hilbert_ideal_dirac.jpg")
    # plt.close()

    x3 = np.sin(2*np.pi*0.5 * t)
    # x3 -= x3.mean()
    y3 = hilbert(x3)
    x3 /= np.max(np.abs(x3))
    y3 /= np.max(np.abs(y3))
    plt.figure(figsize=(20, 10))
    ax1 = plt.subplot2grid(shape=(2, 4), loc=(0, 0), colspan=2, rowspan=1)
    ax2 = plt.subplot2grid(shape=(2, 4), loc=(1, 0), colspan=2, rowspan=1)
    ax3 = plt.subplot2grid(shape=(2, 4), loc=(0, 2), colspan=2, rowspan=2)
    ax1.set_xlabel("t [s]", size=28)
    ax1.set_ylabel("Acceleration [Normalized]", size=28)
    ax1.set_xticks([0, 60, 120, 180])
    ax1.tick_params(labelsize=24)
    ax2.set_xlabel("t [s]", size=28)
    ax2.set_ylabel("Hilbert [Normalized]", size=28)
    ax2.set_xticks([0, 60, 120, 180])
    ax2.tick_params(labelsize=24)
    ax3.tick_params(labelsize=24)
    ax1.plot(t, x3, c="tab:blue")
    ax2.plot(t, y3, c="tab:blue")
    ax3.scatter(x3[N//1000: N-N//1000], y3[N//1000: N-N//1000], 
                s=32, marker='.', c="tab:green")
    ax3.set_xlim([-1.05, 1.05])
    ax3.set_ylim([-1.05, 1.05])
    ax3.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax3.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax3.set_xlabel("Acceleration [Normalized]", size=28)
    ax3.set_ylabel("Hilbert [Normalized]", size=28)
    plt.tight_layout()
    plt.savefig("./hilbert_ideal_sin.jpg")
    plt.close()

def plot_derivative_ideal():
    N = 50 * 60 * 3  # 10000
    t = np.arange(N) * 0.02
    dt = 0.02

    # # step
    # x1 = np.zeros(N)
    # x1[N // 4 * 3:] = 10000
    # # x1 -= x1.mean()
    # y1 = calc_derivative(x1, dt)
    # x1 /= np.max(np.abs(x1))
    # y1 /= np.max(np.abs(y1))
    # plt.figure(figsize=(20, 10))
    # ax1 = plt.subplot2grid(shape=(2, 4), loc=(0, 0), colspan=2, rowspan=1)
    # ax2 = plt.subplot2grid(shape=(2, 4), loc=(1, 0), colspan=2, rowspan=1)
    # ax3 = plt.subplot2grid(shape=(2, 4), loc=(0, 2), colspan=2, rowspan=2)
    # ax1.set_xlabel("t [s]", size=28)
    # ax1.set_ylabel("Acceleration [Normalized]", size=28)
    # ax1.tick_params(labelsize=24)
    # ax2.set_xlabel("t [s]", size=28)
    # ax2.set_ylabel("Derivative [Normalized]", size=28)
    # ax2.tick_params(labelsize=24)
    # ax3.tick_params(labelsize=24)
    # ax1.plot(t, x1, c="tab:blue")
    # ax2.plot(t, y1, c="tab:blue")
    # ax3.scatter(x1, y1, s=1, marker='*')
    # ax3.set_xlim([-1.05, 1.05])
    # ax3.set_ylim([-1.05, 1.05])
    # plt.tight_layout()
    # plt.savefig("./derivative_ideal_step.jpg")
    # plt.close()

    # # dirac
    # x2 = np.zeros(N)
    # x2[N // 4 * 3: N // 4 * 3 + 10] = 10000
    # # x2 -= x2.mean()
    # y2= calc_derivative(x2, dt)
    # x2 /= np.max(np.abs(x2))
    # y2 /= np.max(np.abs(y2))
    # plt.figure(figsize=(20, 10))
    # ax1 = plt.subplot2grid(shape=(2, 4), loc=(0, 0), colspan=2, rowspan=1)
    # ax2 = plt.subplot2grid(shape=(2, 4), loc=(1, 0), colspan=2, rowspan=1)
    # ax3 = plt.subplot2grid(shape=(2, 4), loc=(0, 2), colspan=2, rowspan=2)
    # ax1.set_xlabel("t [s]", size=28)
    # ax1.set_ylabel("Acceleration [Normalized]", size=28)
    # ax1.tick_params(labelsize=24)
    # ax2.set_xlabel("t [s]", size=28)
    # ax2.set_ylabel("Derivative [Normalized]", size=28)
    # ax2.tick_params(labelsize=24)
    # ax3.tick_params(labelsize=24)
    # ax1.plot(t, x2, c="tab:blue")
    # ax2.plot(t, y2, c="tab:blue")
    # ax3.scatter(x2, y2, s=1, marker='*')
    # ax3.set_xlim([-1.05, 1.05])
    # ax3.set_ylim([-1.05, 1.05])
    # plt.tight_layout()
    # plt.savefig("./derivative_ideal_dirac.jpg")
    # plt.close()

    # sine
    x3 = np.sin(2*np.pi*0.5 * t)
    # x3 -= x3.mean()
    y3 = calc_derivative(x3, dt)
    x3 /= np.max(np.abs(x3))
    y3 /= np.max(np.abs(y3))
    plt.figure(figsize=(20, 10))
    ax1 = plt.subplot2grid(shape=(2, 4), loc=(0, 0), colspan=2, rowspan=1)
    ax2 = plt.subplot2grid(shape=(2, 4), loc=(1, 0), colspan=2, rowspan=1)
    ax3 = plt.subplot2grid(shape=(2, 4), loc=(0, 2), colspan=2, rowspan=2)
    ax1.set_xlabel("t [s]", size=28)
    ax1.set_ylabel("Acceleration [Normalized]", size=28)
    ax1.set_xticks([0, 60, 120, 180])
    ax1.tick_params(labelsize=24)
    ax2.set_xlabel("t [s]", size=28)
    ax2.set_ylabel("Derivative [Normalized]", size=28)
    ax2.set_xticks([0, 60, 120, 180])
    ax2.tick_params(labelsize=24)
    ax3.tick_params(labelsize=24)
    ax1.plot(t, x3, c="tab:blue")
    ax2.plot(t, y3, c="tab:blue")
    ax3.scatter(x3, y3, s=32, marker='.', c="tab:green")
    ax3.set_xlim([-1.05, 1.05])
    ax3.set_ylim([-1.05, 1.05])
    ax3.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax3.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax3.set_xlabel("Acceleration [Normalized]", size=28)
    ax3.set_ylabel("Derivative [Normalized]", size=28)
    plt.tight_layout()
    plt.savefig("./derivative_ideal_sin.jpg")
    plt.close()

def plot_hilbert_real():
    mark = ['og', 'ob', 'om', 'oy']

    # # dirac
    # x = loadmat("./2021-04-24 16-VIC.mat")["data"][:, 0]
    # N = x.shape[0] // 20
    # t = np.arange(N) * 0.02
    # x1 = x[5*N: 6*N]
    # # x1 -= x1.mean()
    # y1 = hilbert(x1)
    # x1 /= np.max(np.abs(x1))
    # y1 /= np.max(np.abs(y1))
    # plt.figure(figsize=(20, 10))
    # ax1 = plt.subplot2grid(shape=(2, 4), loc=(0, 0), colspan=2, rowspan=1)
    # ax2 = plt.subplot2grid(shape=(2, 4), loc=(1, 0), colspan=2, rowspan=1)
    # ax3 = plt.subplot2grid(shape=(2, 4), loc=(0, 2), colspan=2, rowspan=2)
    # ax1.set_xlabel("t [s]", size=28)
    # ax1.set_ylabel("Acceleration [Normalized]", size=28)
    # ax1.tick_params(labelsize=24)
    # ax2.set_xlabel("t [s]", size=28)
    # ax2.set_ylabel("Hilbert [Normalized]", size=28)
    # ax2.tick_params(labelsize=24)
    # ax3.tick_params(labelsize=24)
    # ax1.plot(t, x1, c="tab:blue")
    # ax2.plot(t, y1, c="tab:blue")
    # # ax3.scatter(x1, y1, s=1, marker='*')
    # ax3.set_xlim([-1.05, 1.05])
    # ax3.set_ylim([-1.05, 1.05])
    # model1 = DBSCAN(eps=0.1, min_samples=N//100)
    # result = model1.fit_predict(x1.reshape(-1, 1))
    # # result = np.zeros_like(x1, dtype="int32")
    # for i, (x0, y0) in enumerate(zip(x1, y1)):
    #     ax3.plot(x0, y0, mark[result[i]])
    # plt.tight_layout()
    # plt.savefig("./hilbert_real_step.jpg")
    # plt.close()

    # # step
    # x2 = x[N: 2*N]
    # # x2 -= x2.mean()
    # y2= hilbert(x2)
    # x2 /= np.max(np.abs(x2))
    # y2 /= np.max(np.abs(y2))
    # plt.figure(figsize=(20, 10))
    # ax1 = plt.subplot2grid(shape=(2, 4), loc=(0, 0), colspan=2, rowspan=1)
    # ax2 = plt.subplot2grid(shape=(2, 4), loc=(1, 0), colspan=2, rowspan=1)
    # ax3 = plt.subplot2grid(shape=(2, 4), loc=(0, 2), colspan=2, rowspan=2)
    # ax1.set_xlabel("t [s]", size=28)
    # ax1.set_ylabel("Acceleration [Normalized]", size=28)
    # ax1.tick_params(labelsize=24)
    # ax2.set_xlabel("t [s]", size=28)
    # ax2.set_ylabel("Hilbert [Normalized]", size=28)
    # ax2.tick_params(labelsize=24)
    # ax3.tick_params(labelsize=24)
    # ax1.plot(t, x2, c="tab:blue")
    # ax2.plot(t, y2, c="tab:blue")
    # # ax3.scatter(x2, y2, s=1, marker='*')
    # ax3.set_xlim([-1.05, 1.05])
    # ax3.set_ylim([-1.05, 1.05])
    # model2 = DBSCAN(eps=0.1, min_samples=N//100)
    # result = model2.fit_predict(x2.reshape(-1, 1))
    # for i, (x0, y0) in enumerate(zip(x2, y2)):
    #     ax3.plot(x0, y0, mark[result[i]])
    # plt.tight_layout()
    # plt.savefig("./hilbert_real_dirac.jpg")
    # plt.close()

    # sine
    x = loadmat("./2021-04-23 10-VIC.mat")["data"][:, 1]
    N = x.shape[0] // 20
    t = np.arange(N) * 0.02
    x3 = x[5*N: 6*N]
    # x3 -= x3.mean()
    y3 = hilbert(x3)
    x3 /= np.max(np.abs(x3))
    y3 /= np.max(np.abs(y3))
    plt.figure(figsize=(20, 10))
    ax1 = plt.subplot2grid(shape=(2, 4), loc=(0, 0), colspan=2, rowspan=1)
    ax2 = plt.subplot2grid(shape=(2, 4), loc=(1, 0), colspan=2, rowspan=1)
    ax3 = plt.subplot2grid(shape=(2, 4), loc=(0, 2), colspan=2, rowspan=2)
    ax1.set_xlabel("t [s]", size=28)
    ax1.set_ylabel("Acceleration [Normalized]", size=28)
    ax1.set_xticks([0, 60, 120, 180])
    ax1.tick_params(labelsize=24)
    ax2.set_xlabel("t [s]", size=28)
    ax2.set_ylabel("Hilbert [Normalized]", size=28)
    ax2.set_xticks([0, 60, 120, 180])
    ax2.tick_params(labelsize=24)
    ax3.tick_params(labelsize=24)
    ax1.plot(t, x3, c="tab:blue")
    ax2.plot(t, y3, c="tab:blue")
    # ax3.scatter(x3[:N//1000], y3[:N//1000], s=32, marker='*', c='b')  # key error
    # ax3.scatter(x3[N-N//1000:], y3[N-N//1000:], s=32, marker='*', c='r')  # minor influence
    ax3.scatter(x3[N//1000: N-N//1000:], y3[N//1000: N-N//1000:], s=32, marker='.', c='g')
    # model3 = DBSCAN(eps=0.1, min_samples=N//100)
    # result = model3.fit_predict(x3.reshape(-1, 1))
    # for i, (x0, y0) in enumerate(zip(x3, y3)):
    #     # if np.abs(x0) <= 0.2 and np.abs(y0) <= 0.2:
    #     #     continue
    #     ax3.plot(x0, y0, mark[result[i]])
    # plot inner radius and outer radius
    x3_ = x3[N//1000: N-N//1000:]
    y3_ = y3[N//1000: N-N//1000:]
    R = np.sqrt(x3_ ** 2 + y3_ ** 2)
    min_index = np.argmin(R)
    max_index = np.argmax(R)
    ax3.arrow(0, 0, x3_[min_index], y3_[min_index], width=0.01, length_includes_head=True, 
              head_width=0.05,head_length=0.1, fc='r', ec='r')
    ax3.arrow(0, 0, x3_[max_index], y3_[max_index], width=0.01, length_includes_head=True, 
              head_width=0.05,head_length=0.1, fc='r', ec='r')
    ax3.text(x3_[min_index]/2, y3_[min_index]/2, "$R_1$", va="center", ha="center", fontsize=48, color='r')
    ax3.text(x3_[max_index]/2, y3_[max_index]/2, "$R_2$", va="center", ha="center", fontsize=48, color='r')
    ax3.set_xlim([-1.05, 1.05])
    ax3.set_ylim([-1.05, 1.05])
    ax3.set_xlabel("Acceleration [Normalized]", size=28)
    ax3.set_ylabel("Hilbert [Normalized]", size=28)
    ax3.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax3.set_yticks([-1, -0.5, 0, 0.5, 1])
    plt.tight_layout()
    plt.savefig("./hilbert_real_sin.jpg")
    plt.close()

def plot_derivative_real():
    mark = ['og', 'ob', 'om', 'oy']
    dt = 0.02

    # # dirac
    # x = loadmat("./2021-04-24 16-VIC.mat")["data"][:, 0]
    # N = x.shape[0] // 20
    # t = np.arange(N) * dt
    # x1 = x[5*N: 6*N]
    # # x1 -= x1.mean()
    # y1 = calc_derivative(x1, dt)
    # x1 /= np.max(np.abs(x1))
    # y1 /= np.max(np.abs(y1))
    # plt.figure(figsize=(20, 10))
    # ax1 = plt.subplot2grid(shape=(2, 4), loc=(0, 0), colspan=2, rowspan=1)
    # ax2 = plt.subplot2grid(shape=(2, 4), loc=(1, 0), colspan=2, rowspan=1)
    # ax3 = plt.subplot2grid(shape=(2, 4), loc=(0, 2), colspan=2, rowspan=2)
    # ax1.set_xlabel("t [s]", size=28)
    # ax1.set_ylabel("Acceleration [Normalized]", size=28)
    # ax1.tick_params(labelsize=24)
    # ax2.set_xlabel("t [s]", size=28)
    # ax2.set_ylabel("Derivative [Normalized]", size=28)
    # ax2.tick_params(labelsize=24)
    # ax3.tick_params(labelsize=24)
    # ax1.plot(t, x1, c="tab:blue")
    # ax2.plot(t, y1, c="tab:blue")
    # # ax3.scatter(x1, y1, s=1, marker='*')
    # ax3.set_xlim([-1.05, 1.05])
    # ax3.set_ylim([-1.05, 1.05])
    # model1 = DBSCAN(eps=0.1, min_samples=N//100)
    # result = model1.fit_predict(x1.reshape(-1, 1))
    # for i, (x0, y0) in enumerate(zip(x1, y1)):
    #     ax3.plot(x0, y0, mark[result[i]])
    # plt.tight_layout()
    # plt.savefig("./derivative_real_step.jpg")
    # plt.close()

    # # step
    # x2 = x[N: 2*N]
    # # x2 -= x2.mean()
    # y2= calc_derivative(x2, dt)
    # x2 /= np.max(np.abs(x2))
    # y2 /= np.max(np.abs(y2))
    # plt.figure(figsize=(20, 10))
    # ax1 = plt.subplot2grid(shape=(2, 4), loc=(0, 0), colspan=2, rowspan=1)
    # ax2 = plt.subplot2grid(shape=(2, 4), loc=(1, 0), colspan=2, rowspan=1)
    # ax3 = plt.subplot2grid(shape=(2, 4), loc=(0, 2), colspan=2, rowspan=2)
    # ax1.set_xlabel("t [s]", size=28)
    # ax1.set_ylabel("Acceleration [Normalized]", size=28)
    # ax1.tick_params(labelsize=24)
    # ax2.set_xlabel("t [s]", size=28)
    # ax2.set_ylabel("Derivative [Normalized]", size=28)
    # ax2.tick_params(labelsize=24)
    # ax3.tick_params(labelsize=24)
    # ax1.plot(t, x2, c="tab:blue")
    # ax2.plot(t, y2, c="tab:blue")
    # # ax3.scatter(x2, y2, s=1, marker='*')
    # ax3.set_xlim([-1.05, 1.05])
    # ax3.set_ylim([-1.05, 1.05])
    # model2 = DBSCAN(eps=0.1, min_samples=N//100)
    # result = model2.fit_predict(x2.reshape(-1, 1))
    # for i, (x0, y0) in enumerate(zip(x2, y2)):
    #     ax3.plot(x0, y0, mark[result[i]])
    # plt.tight_layout()
    # plt.savefig("./derivative_real_dirac.jpg")
    # plt.close()

    # sine
    x = loadmat("./2021-04-23 10-VIC.mat")["data"][:, 1]
    N = x.shape[0] // 20
    t = np.arange(N) * 0.02
    x3 = x[5*N: 6*N]
    # x3 -= x3.mean()
    y3 = calc_derivative(x3, dt)
    x3 /= np.max(np.abs(x3))
    y3 /= np.max(np.abs(y3))
    plt.figure(figsize=(20, 10))
    ax1 = plt.subplot2grid(shape=(2, 4), loc=(0, 0), colspan=2, rowspan=1)
    ax2 = plt.subplot2grid(shape=(2, 4), loc=(1, 0), colspan=2, rowspan=1)
    ax3 = plt.subplot2grid(shape=(2, 4), loc=(0, 2), colspan=2, rowspan=2)
    ax1.set_xlabel("t [s]", size=28)
    ax1.set_ylabel("Acceleration [Normalized]", size=28)
    ax1.tick_params(labelsize=24)
    ax1.set_xticks([0, 60, 120, 180])
    ax2.set_xlabel("t [s]", size=28)
    ax2.set_ylabel("Derivative [Normalized]", size=28)
    ax2.set_xticks([0, 60, 120, 180])
    ax2.tick_params(labelsize=24)
    ax3.tick_params(labelsize=24)
    ax1.plot(t, x3, c="tab:blue")
    ax2.plot(t, y3, c="tab:blue")
    # ax3.scatter(x3[:N//1000], y3[:N//1000], s=32, marker='*', c='b')  # key error
    # ax3.scatter(x3[N-N//1000:], y3[N-N//1000:], s=32, marker='*', c='r')  # minor influence
    ax3.scatter(x3[N//1000: N-N//1000:], y3[N//1000: N-N//1000:], s=32, marker='.', c='g')
    # model3 = DBSCAN(eps=0.1, min_samples=N//100)
    # result = model3.fit_predict(x3.reshape(-1, 1))
    # for i, (x0, y0) in enumerate(zip(x3, y3)):
    #     ax3.plot(x0, y0, mark[result[i]])
    ax3.set_xlim([-1.05, 1.05])
    ax3.set_ylim([-1.05, 1.05])
    # plot inner radius and outer radius
    R = np.sqrt(x3 ** 2 + y3 ** 2)
    min_index = np.argmin(R)
    max_index = np.argmax(R)
    ax3.arrow(0, 0, x3[min_index], y3[min_index], width=0.01, length_includes_head=True, 
              head_width=0.05,head_length=0.1, fc='r', ec='r')
    ax3.arrow(0, 0, x3[max_index], y3[max_index], width=0.01, length_includes_head=True, 
              head_width=0.05,head_length=0.1, fc='r', ec='r')
    ax3.text(x3[min_index]/2, y3[min_index]/2, "$R_1$", va="center", ha="center", fontsize=48, color='r')
    ax3.text(x3[max_index]/2, y3[max_index]/2, "$R_2$", va="center", ha="center", fontsize=48, color='r')
    ax3.set_xlabel("Acceleration [Normalized]", size=28)
    ax3.set_ylabel("Derivative [Normalized]", size=28)
    ax3.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax3.set_yticks([-1, -0.5, 0, 0.5, 1])
    plt.tight_layout()
    plt.savefig("./derivative_real_sin.jpg")
    plt.close()

if __name__ == "__main__":
    plot_hilbert_ideal()
    plot_derivative_ideal()
    plot_hilbert_real()
    plot_derivative_real()
