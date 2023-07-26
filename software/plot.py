import os
import numpy as np
from scipy.io import loadmat
from scipy.fftpack import fft, fftfreq
import matplotlib.pyplot as plt
from calc import calc_derivative, calc_feature
plt.rc('font', family='Times New Roman')
plt.rc('text', usetex = True)

def plot_acc_time_history(pic_path, acc, fs, RMS):
    ''' 
    purpose: plot acceleration time history, frequency spectrum, derivative complex signal 
    params: pic_path - picture path, includes directory and name
            acc - time history series of acceleration
            fs - sampling frequency
            RMS - root mean square of acc 
    return: None
    '''

    Ts = 1 / fs
    num_sample = acc.shape[0]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(Ts * np.arange(num_sample), acc, c="tab:green")
    ax.set_xticks(60 * np.arange(0, 6))
    plt.tick_params(labelsize=16)
    ax.set_xlabel("Time [s]", size=24)
    ax.set_ylabel("Acceleration [mg]", size=24)
    ax.text(0.5, 0.5, "RMS = %.4f" % RMS, fontsize=24, color='k', 
            transform=ax.transAxes, va="center", ha="center")
    plt.savefig(f"{pic_path}_acc_time_history.jpg", bbox_inches="tight", pad_inches=0.1)
    plt.close()

def plot_freq_spec(pic_path, acc, fs, max_A_freq, fund_freq=0.3, threshold=None):
    ''' 
    purpose: plot acceleration time history, frequency spectrum, derivative complex signal 
    params: pic_path - picture path, includes directory and name
            acc - time history series of acceleration
            fs - sampling frequency
            max_A_freq - frequency with the largest amplitude
            fund_freq - fundamental frequency (design value or the previous value)
            threshold - coefficient of the high-pass filter    
    return: None
    '''

    if not threshold:
        threshold = 0.5 * fund_freq

    Ts = 1 / fs
    num_sample = acc.shape[0]

    spec = fft(acc)
    f = fftfreq(num_sample, Ts)  # f.max() == fs/2 : Nesquite's sampling theory
    half_f = f[:(num_sample+1) // 2]
    half_spec = np.abs(spec[:(num_sample+1) // 2])
    half_spec[0] /= 2
    wave_filter = (half_f > threshold)
    half_spec *= wave_filter
    half_spec /= half_spec.max()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(half_f, half_spec, c="tab:orange", alpha=0.5)
    plt.tick_params(labelsize=16)
    ax.set_ylim([-0.1, 1.1])
    ax.set_yticks([0, 1])
    ax.set_xlim(-1, 0.8 * fs/2 + 1)
    ax.set_xticks(np.arange(1, 0.8 * fs/2 + 1, 2))
    ax.set_xlabel("Frequency [Hz]", size=24)
    ax.set_ylabel("Amplitude", size=24)
    ax.scatter([max_A_freq], [1.], c="tab:purple", s=64, marker='*')
    ax.text(max_A_freq, 1., "%.3f" % max_A_freq, color="tab:purple", 
             fontsize=32, va="bottom", ha="center")
    plt.savefig(f"{pic_path}_freq_spec.jpg", bbox_inches="tight", pad_inches=0.1)
    plt.close()

def plot_deri_signal(pic_path, acc, fs, HCD):
    ''' 
    purpose: plot derivative complex signal 
    params: pic_path - picture path, includes directory and name
            acc - time history series of acceleration
            fs - sampling frequency
            HCD - hollow coefficient of Derivative analytical signal of acc 
    return: None
    '''

    Ts = 1 / fs

    dt = Ts
    unit_margin = 1.02
    x = acc / np.abs(acc).max()
    y = calc_derivative(x, dt)
    y /= np.abs(y).max()

    fig, ax = plt.subplots(figsize=(10, 10))
    plt.axis("equal")
    plt.scatter(x, y, c='tab:olive')
    plt.tick_params(labelsize=16)
    ax.text(0.5, 0.5, "HCD = %.4f" % HCD, fontsize=24, color='k', 
            transform=ax.transAxes, va="center", ha="center")
    ax.set_xlim([-unit_margin, unit_margin])
    ax.set_ylim([-unit_margin, unit_margin])
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xlabel("Origin", size=24)
    ax.set_ylabel("Derivative", size=24)
    plt.savefig(f"{pic_path}_deri_scatter.jpg", bbox_inches="tight", pad_inches=0.1)
    plt.close()

def plot_ath_spec_deri(pic_path, acc, fs, RMS, HCD, max_A_freq, fund_freq=0.3, threshold=None):
    ''' 
    purpose: plot acceleration time history, frequency spectrum, derivative complex signal 
    params: pic_path - picture path, includes directory and name
            acc - time history series of acceleration
            fs - sampling frequency
            RMS - root mean square of acc
            HCD - hollow coefficient of Derivative analytical signal of acc
            max_A_freq - frequency with the largest amplitude
            fund_freq - fundamental frequency (design value or the previous value)
            threshold - coefficient of the high-pass filter    
    return: None
    '''

    # time history
    plot_acc_time_history(pic_path, acc, fs, RMS)
    
    # frequency spectrum
    plot_freq_spec(pic_path, acc, fs, max_A_freq, fund_freq, threshold)

    # derivative
    plot_deri_signal(pic_path, acc, fs, HCD)

def main():
    # test calc_feature
    mat_file = os.path.join(".", "2021-04-23 10-VIC.mat")
    total_acc = loadmat(mat_file)["data"][:, 1]
    total_length = total_acc.shape[0]
    num_sample = total_length // 12
    acc = total_acc[num_sample: num_sample * 2]
    fs = 50
    app_fund_freq = 0.59
    m_bar = 1
    L = 1
    error_flag, RMS, HCD, fund_freq, max_A_freq, max_acc, cable_force = \
            calc_feature(acc, fs, app_fund_freq, m_bar, L, threshold=None)
    print(error_flag, RMS, HCD, fund_freq, max_A_freq, max_acc, cable_force)

    # test calc_plot_ath_spec_der
    pic_path = "./2021-04-23-05min-10min-channel-2"
    plot_ath_spec_deri(pic_path, acc, fs, RMS, HCD, max_A_freq, fund_freq)
    
if __name__ == "__main__":
    main()
