import numpy as np
from scipy.io import loadmat
from scipy.fftpack import fft, fftfreq, hilbert
import matplotlib.pyplot as plt

plt.rc('font', family='Times New Roman')
plt.rc('text', usetex = True)

def calc_derivative(arr, dt): 
    ''' dx/dt = (x[t+dt] - x[t-dt]) / (2*dt) '''
    a = np.zeros_like(arr)
    a[1 : -1] = (arr[2:] - arr[:-2]) / (2 * dt)
    a[0] = (arr[1] - arr[0]) / dt
    a[-1] = (arr[-1] - arr[-2]) / dt
    return a

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

def plot_time_spec_hil_deri(a, jpg_name, cut_rate=0.):
    fs = 50
    Ts = 1 / fs
    interval = a.shape[0]

    plt.figure(figsize=(15, 10))
    ax_time = plt.subplot2grid(shape=(2, 3), loc=(0, 0), colspan=2, rowspan=1)
    ax_spec = plt.subplot2grid(shape=(2, 3), loc=(1, 0), colspan=2, rowspan=1)
    ax_hilb = plt.subplot2grid(shape=(2, 3), loc=(0, 2), colspan=1, rowspan=1)
    ax_deri = plt.subplot2grid(shape=(2, 3), loc=(1, 2), colspan=1, rowspan=1)

    # time history
    RMS = np.sqrt(np.sum(a ** 2) / interval)
    ax_time.plot(Ts * np.arange(interval), a, c="tab:cyan")
    ax_time.set_xlim(-1, interval*Ts + 1)
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
    threshold=0.4
    app_fund=0.5
    spec = fft(a)
    f = fftfreq(interval, Ts)  # f.max() == fs/2 : Nesquite's sampling theory
    half_f = f[:(interval+1) // 2]
    half_spec = np.abs(spec[:(interval+1) // 2])
    half_spec[0] /= 2
    wave_filter = (half_f > threshold)
    half_spec *= wave_filter
    half_spec /= half_spec.max()
    ax_spec.plot(half_f, half_spec, c="tab:green")
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
    ax_spec.text(0.75, 0.50, "PRS = %.4f" % PRS, fontsize=27, color='k', 
            transform=ax_spec.transAxes, va="center", ha="center")
    ax_spec.scatter([F1, F2], [A1, A2], c="tab:red", s=128, marker='*')

    # H.T.
    N = a.shape[0]
    n = int(N * cut_rate)
    x = a[n: N-n]
    y = hilbert(a)[n: N-n]
    x /= np.abs(x).max()
    y /= np.abs(y).max()
    CR = np.mean(x)
    CI = np.mean(y)

    # CCH
    alpha = 0.
    R = np.sqrt((x-CR) ** 2 + (y-CI) ** 2)
    w = np.exp(- alpha * R)  # alpha = 0, 1, 2, 3...
    R_mean = np.sum(w * R) / np.sum(w)
    R_std = np.sqrt(np.sum(w * (R-R_mean)**2) / np.sum(w))
    CCH = R_mean / R_std
    ax_hilb.axis("equal")
    ax_hilb.scatter(x, y, c='tab:orange', s=1)
    ax_hilb.text(0.5, 0.5, "CCH=\n%.4f" % CCH, fontsize=27, color='k', 
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
    y = calc_derivative(x, dt)
    y /= np.abs(y).max()
    CR = np.mean(x)
    CI = np.mean(y)

    # CCD
    alpha = 0.
    R = np.sqrt((x-CR) ** 2 + (y-CI) ** 2)
    w = np.exp(- alpha * R)  # alpha = 0, 1, 2, 3...
    R_mean = np.sum(w * R) / np.sum(w)
    R_std = np.sqrt(np.sum(w * (R-R_mean)**2) / np.sum(w))
    CCD = R_mean / R_std
    ax_deri.axis("equal")
    ax_deri.scatter(x, y, c='tab:blue', s=1)
    ax_deri.text(0.5, 0.5, "CCD=\n%.4f" % CCD, fontsize=27, color='k', 
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

def plot_hilbert():
    mark = ['og', 'ob', 'om', 'oy']

    # sine
    x = loadmat("./2021-04-23 10-VIC.mat")["data"][:, 1]
    N = x.shape[0] // 20  # 3 min
    t = np.arange(N) * 0.02
    x3 = x[5*N: 6*N]
    y3 = hilbert(x3)
    x3 /= np.max(np.abs(x3))
    y3 /= np.max(np.abs(y3))
    plt.figure(figsize=(30, 11))
    ax1 = plt.subplot2grid(shape=(1, 3), loc=(0, 0), colspan=2, rowspan=1)
    ax3 = plt.subplot2grid(shape=(1, 3), loc=(0, 2), colspan=1, rowspan=1)
    ax1.set_xlabel("t [s]", size=48)
    ax1.set_ylabel("Normalized Signal", size=48)
    ax1.set_xticks([0, 0.5, 1, 1.5, 2.0])
    ax1.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax3.axis("equal")
    ax1.tick_params(labelsize=36)
    ax3.tick_params(labelsize=36)
    ax1.plot(t[:N//90], x3[:N//90], c="tab:cyan", linestyle='-', linewidth=3)  # N: 180 s, N//90: 2 s
    ax1.plot(t[:N//90], y3[:N//90], c="tab:green", linestyle='--', linewidth=3)
    ax1.legend(["Origin", "Hilbert"], fontsize=32, loc="upper right")
    ax3.scatter(x3[N//1000: N-N//1000:], y3[N//1000: N-N//1000:], s=32, marker='.', c="tab:orange")
    x3_ = x3[N//1000: N-N//1000:]
    y3_ = y3[N//1000: N-N//1000:]
    R = np.sqrt(x3_ ** 2 + y3_ ** 2)
    min_index = np.argmin(R)
    max_index = np.argmax(R)
    ax3.arrow(0, 0, x3_[min_index], y3_[min_index], width=0.01, length_includes_head=True, 
              head_width=0.05,head_length=0.1, fc="tab:blue", ec="tab:blue")
    ax3.arrow(0, 0, x3_[max_index], y3_[max_index], width=0.01, length_includes_head=True, 
              head_width=0.05,head_length=0.1, fc="tab:blue", ec="tab:blue")
    ax3.text(x3_[min_index]/2, y3_[min_index]/2, r"$R_{min}$", va="center", ha="center", fontsize=48, color="tab:blue")
    ax3.text(x3_[max_index]/2, y3_[max_index]/2, r"$R_{max}$", va="center", ha="center", fontsize=48, color="tab:blue")
    ax3.set_xlim([-1.05, 1.05])
    ax3.set_ylim([-1.05, 1.05])
    ax3.set_xlabel("Origin [Normalized]", size=48)
    ax3.set_ylabel("Hilbert [Normalized]", size=48)
    ax3.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax3.set_yticks([-1, -0.5, 0, 0.5, 1])
    plt.tight_layout()
    plt.savefig("./HCH.jpg", bbox_inches="tight", pad_inches=0.05)
    plt.close()

def plot_derivative():
    mark = ['og', 'ob', 'om', 'oy']
    dt = 0.02

    # sine
    x = loadmat("./2021-04-23 10-VIC.mat")["data"][:, 1]
    N = x.shape[0] // 20
    t = np.arange(N) * 0.02
    x3 = x[5*N: 6*N]
    y3 = calc_derivative(x3, dt)
    x3 /= np.max(np.abs(x3))
    y3 /= np.max(np.abs(y3))
    plt.figure(figsize=(30, 11))
    ax1 = plt.subplot2grid(shape=(1, 3), loc=(0, 0), colspan=2, rowspan=1)
    ax3 = plt.subplot2grid(shape=(1, 3), loc=(0, 2), colspan=1, rowspan=1)
    ax1.set_xlabel("t [s]", size=48)
    ax1.set_ylabel("Normalized Signal", size=48)
    ax1.set_xticks([0, 0.5, 1, 1.5, 2.0])
    ax1.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax3.axis("equal")
    ax1.tick_params(labelsize=36)
    ax3.tick_params(labelsize=36)
    ax1.plot(t[:N//90], x3[:N//90], c="tab:cyan", linestyle='-', linewidth=3)  # N: 180 s, N//90: 2 s
    ax1.plot(t[:N//90], y3[:N//90], c="tab:green", linestyle='--', linewidth=3)
    ax1.legend(["Origin", "Derivative"], fontsize=32, loc="upper right")
    # ax3.scatter(x3[:N//1000], y3[:N//1000], s=32, marker='*', c='b')  # key error
    # ax3.scatter(x3[N-N//1000:], y3[N-N//1000:], s=32, marker='*', c='r')  # minor influence
    ax3.scatter(x3[N//1000: N-N//1000:], y3[N//1000: N-N//1000:], s=32, marker='.', c="tab:orange")
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
              head_width=0.05,head_length=0.1, fc="tab:blue", ec="tab:blue")
    ax3.arrow(0, 0, x3[max_index], y3[max_index], width=0.01, length_includes_head=True, 
              head_width=0.05,head_length=0.1, fc="tab:blue", ec="tab:blue")
    ax3.text(x3[min_index]/2, y3[min_index]/2, r"$R_{min}$", va="center", ha="center", fontsize=48, color="tab:blue")
    ax3.text(x3[max_index]/2, y3[max_index]/2, r"$R_{max}$", va="center", ha="center", fontsize=48, color="tab:blue")
    ax3.set_xlabel("Origin [Normalized]", size=48)
    ax3.set_ylabel("Derivative [Normalized]", size=48)
    ax3.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax3.set_yticks([-1, -0.5, 0, 0.5, 1])
    plt.tight_layout()
    plt.savefig("./HCD.jpg", bbox_inches="tight", pad_inches=0.05)
    plt.close()

def plot_PRS():
    mark = ['og', 'ob', 'om', 'oy']
    fs = 50
    Ts = 1 / fs

    # sine
    x = loadmat("./2021-04-23 10-VIC.mat")["data"][:, 1]
    interval = x.shape[0] // 20
    t = np.arange(interval) * 0.02
    x3 = x[5*interval: 6*interval]

    # spectrum
    threshold = 0.4
    app_fund = 0.5
    spec = fft(x3)
    f = fftfreq(interval, Ts)  # f.max() == fs/2 : Nesquite's sampling theory
    half_f = f[:(interval+1) // 2]
    half_spec = np.abs(spec[:(interval+1) // 2])
    half_spec[0] /= 2
    wave_filter = (half_f > threshold)
    half_spec *= wave_filter
    half_spec /= half_spec.max()
    fig, ax_spec = plt.subplots(1, 1, figsize=(20, 8))
    ax_spec.plot(half_f, half_spec, c="tab:green", linewidth=3)
    ax_spec.set_ylim([-0.1, 1.1])
    ax_spec.set_xticks(range(0, 22, 2))
    ax_spec.set_yticks([0, 1])
    ax_spec.set_xlim(0, 20)
    ax_spec.tick_params(labelsize=36)
    ax_spec.set_xlabel("Frequency [Hz]", fontsize=48, color='k')
    ax_spec.set_ylabel("Amplitude", fontsize=48, color='k')
    # calc PRS
    FSM_interval = int(1.5 * app_fund / half_f[1])
    peaks = np.array(find_peak(half_spec, FSM_interval))
    sorted_peaks = sorted(peaks, key=lambda i: half_spec[i])
    F1 = half_f[sorted_peaks[-1]]
    F2 = half_f[sorted_peaks[-2]]
    A1 = half_spec[sorted_peaks[-1]]
    A2 = half_spec[sorted_peaks[-2]]
    PRS = A2 / A1
    # ax_spec.text(0.7, 0.7, "PRS=\n%.4f" % PRS, fontsize=27, color='k', 
    #         transform=ax_spec.transAxes, va="center", ha="center")
    ax_spec.scatter([F1, F2], [A1, A2], c="tab:red", s=512, marker='*')
    for (x, y, s) in zip([F1, F2], [A1, A2], ["$H_1$", "$H_2$"]):
        ax_spec.text(x+0.8, y, s, c="tab:red", fontsize=48, 
                     va="center", ha="center")

    plt.tight_layout()
    plt.savefig("./PRS.jpg", bbox_inches="tight", pad_inches=0.05)
    plt.close()

def plot_WHCD():
    dt = 0.02
    theta = np.linspace(0, 2*np.pi, 1000, endpoint=False)
    alpha = 0.  # (0, 1]
    x = loadmat("./2021-04-23 10-VIC.mat")["data"][:, 1]
    N = x.shape[0] // 20  # samples in 3 minutes

    for x_, pic_name in zip([x[5*N: 6*N], x[0*N: 1*N]], ["CCD-VIV.jpg", "CCD-non-VIV.jpg"]):
        x1 = x_
        y1 = calc_derivative(x1, dt)
        x1 /= np.max(np.abs(x1))
        y1 /= np.max(np.abs(y1))
        R = np.sqrt(x1 ** 2 + y1 ** 2)
        w = np.exp(- alpha * R)  # alpha = 0, 1, 2, 3...
        R_mean = np.sum(w * R) / np.sum(w)
        R_std = np.sqrt(np.sum(w * (R-R_mean)**2) / np.sum(w))
        CCD = R_mean / R_std
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 10))
        ax1.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax1.set_yticks([-1, -0.5, 0, 0.5, 1])
        ax1.axis("equal")
        ax1.tick_params(labelsize=36)
        ax1.scatter(x1, y1, s=32, marker='.', c="tab:orange", alpha=0.25)
        ax1.set_xlim([-1.05, 1.05])
        ax1.set_ylim([-1.05, 1.05])
        ax1.set_xlabel("Origin [Normalized]", size=48)
        ax1.set_ylabel("Derivative [Normalized]", size=48)
        ax1.arrow(0, 0, R_mean*np.cos(np.pi/4), R_mean*np.sin(np.pi/4), 
                  width=0.01, length_includes_head=True, 
                  head_width=0.05, head_length=0.1, fc="tab:green", ec="tab:green", alpha=0.75)
        ax1.arrow(R_mean, 0, R_std, 0, 
                  width=0.002, length_includes_head=True, 
                  head_width=0.01, head_length=0.02, fc="tab:green", ec="tab:green")
        # ax1.plot([R_mean, R_mean+R_std], [0, 0], c="tab:blue", linestyle="-", linewidth=4)
        ax1.text(R_mean*np.cos(np.pi/4)/2, R_mean*np.sin(np.pi/4)/2, 
                 "$\\mu_{R}$", va="center", ha="center", fontsize=48, color="tab:blue")
        ax1.text(R_mean+0.5*R_std, 0, "$\\sigma_{R}$", 
                 va="bottom", ha="center", fontsize=48, color="tab:blue")
        ax1.plot(R_mean * np.cos(theta), R_mean * np.sin(theta), 
                 c="tab:green", linestyle="--", linewidth=4)
        ax1.plot((R_mean+R_std) * np.cos(theta), (R_mean+R_std) * np.sin(theta), 
                 c="tab:green", linestyle="--", linewidth=4)
        plt.savefig(pic_name, bbox_inches="tight", pad_inches=0.05)
        plt.close()

def main():
    # plot_time_spec_hil_deri
    acc = loadmat("./2021-04-23 10-VIC.mat")["data"]
    N = acc.shape[0]
    n = N // 60
    plot_time_spec_hil_deri(acc[15*n: 25*n, 0], "./non-VIV.jpg")
    plot_time_spec_hil_deri(acc[15*n: 25*n, 1], "./VIV.jpg")

    # plot_hilbert()
    # plot_derivative()
    # plot_PRS()
    # plot_WHCD()

if __name__ == "__main__":
    main()
