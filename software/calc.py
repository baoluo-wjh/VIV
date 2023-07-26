import numpy as np
from scipy.fftpack import fft, fftfreq
import matplotlib.pyplot as plt
plt.rc('font', family='Times New Roman')
plt.rc('text', usetex = True)

# root mean square: RMS
# hollow coefficient of Derivative analytical signal: HCD (SRD)
# peak ratio of spectrum: PRS (SRE), !deprecated
# hollow coefficient of Hilbert analytical signal: HCH (SRA), !deprecated

def calc_derivative(arr, dt): 
    ''' 
    purpose: calculate the numerical derivative of the sequency arr
    params: arr - sequency 
            dt - time num_sample (sampling period) 
    return: der - numerical derivative of the sequency arr
    formula: dx/dt = (x[t+dt] - x[t-dt]) / (2*dt) 
    '''

    der = np.zeros_like(arr)
    der[1 : -1] = (arr[2:] - arr[:-2]) / (2 * dt)
    der[0] = (arr[1] - arr[0]) / dt
    der[-1] = (arr[-1] - arr[-2]) / dt
    return der

def calc_VIV(acc, fs):
    ''' 
    purpose: calculate RMS and SRD of a certain acceleration time history 
    params: acc - time history series of acceleration
            fs - sampling frequency
    return: error_flag = 1 if accelerometer failures happen, else 0
            RMS - root mean square of acc
            HCD - hollow coefficient of Derivative analytical signal of acc
    '''

    Ts = 1 / fs
    np.nan_to_num(acc, copy=False, nan=9999)  # reset NaN as 9999
    num_sample = acc.shape[0]

    if np.abs(acc).max() > 9990 or np.abs(acc).max() < 1:
        # accelerometer failures happen, all == 9999 or all == 0
        error_flag = 1
        return error_flag, 0, 0

    # RMS: time history
    RMS = np.sqrt(np.sum(acc ** 2) / num_sample)
    
    # HCD: derivative
    dt = Ts
    x = acc / np.abs(acc).max()
    y = calc_derivative(x, dt)
    if np.abs(y).max() < 1:
        # accelerometer failures happen, all == const
        error_flag = 1
        return error_flag, 0, 0
    y /= np.abs(y).max()
    center_x = np.mean(x)
    center_y = np.mean(y)
    R = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    HCD = R.min() / R.max()

    # return
    error_flag = 0
    return error_flag, RMS, HCD

def calc_force(acc, fs, app_fund_freq, m_bar, L, threshold=None):
    ''' 
    purpose: calculate RMS and SRD of a certain acceleration time history 
    params: acc - time history series of acceleration
            fs - sampling frequency
            app_fund_freq - appropriate fundamental frequency (design value or the previous value)
            m_bar - linear density of mass of the cable
            L - length of the cable
            threshold - coefficient of the high-pass filter
    return: error_flag = 1 if accelerometer failures happen, else 0
            fund_freq - fundamental frequency
            max_A_freq - frequency with the largest amplitude
            cable_force - tension in cable
    '''

    if not threshold:
        threshold = 0.5 * app_fund_freq

    Ts = 1 / fs
    num_sample = acc.shape[0]

    if np.sum(acc >= 9990) != 0:  # accelerometer failures happen
        error_flag = 1
        return error_flag, 0, 0, 0

    error_flag = 0

    # frequency with the largest amplitude
    spec = fft(acc)
    f = fftfreq(num_sample, Ts)  # f.max() == fs/2 : Nesquite's sampling theory
    half_f = f[:(num_sample+1) // 2]
    half_spec = np.abs(spec[:(num_sample+1) // 2])
    half_spec[0] /= 2
    wave_filter = (half_f > threshold)
    half_spec *= wave_filter
    # half_spec /= half_spec.max()
    max_A_freq = half_f[half_spec.argmax()]

    # fundamental frequency
    n0 = int(round(max_A_freq / app_fund_freq))
    fund_freq = max_A_freq / n0

    # tension in cable
    cable_force = 4 * m_bar * L ** 2 * fund_freq ** 2

    # return
    return error_flag, fund_freq, max_A_freq, cable_force

def calc_feature(acc, fs, app_fund_freq, m_bar, L, threshold=None):
    ''' 
    purpose: calculate RMS and SRD of a certain acceleration time history 
    params: acc - time history series of acceleration
            fs - sampling frequency
            app_fund_freq - appropriate fundamental frequency (design value or the previous value)
            m_bar - linear density of mass of the cable
            L - length of the cable
            threshold - coefficient of the high-pass filter
    return: error_flag = 1 if accelerometer failures happen, else 0
            RMS - root mean square of acc
            HCD - hollow coefficient of Derivative analytical signal of acc
            fund_freq - fundamental frequency
            max_A_freq - frequency with the largest amplitude
            max_acc - maximum acc
            cable_force - tension in cable
    '''

    if not threshold:
        threshold = 0.5 * app_fund_freq

    if np.sum(acc >= 9990) != 0:  # accelerometer failures happen
        error_flag = 1
        return error_flag, 0, 0, 0, 0, 0, 0
    
    # RMS and HCD
    _, RMS, HCD = calc_VIV(acc, fs)

    # force 
    error_flag, fund_freq, max_A_freq, cable_force = \
            calc_force(acc, fs, app_fund_freq, m_bar, L, threshold=None)

    # maximum acc
    max_acc = acc.max()

    # return
    return error_flag, RMS, HCD, fund_freq, max_A_freq, max_acc, cable_force
