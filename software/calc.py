import numpy as np
from scipy.fftpack import fft, fftfreq, hilbert
from scipy.io import loadmat
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
plt.rc('font', family='Times New Roman')
plt.rc('text', usetex = True)

# root mean square: RMS
# peak ratio of spectrum: PRS (SRE), !deprecated
# hollow coefficient of Hilbert analytical signal: HCH (SRA), !deprecated
# hollow coefficient of Derivative analytical signal: HCD (SRD)

def find_peak(arr, interval, min_peak=0.01):
    ''' 
    purpose: find all the local maxima
    params: arr - sequence
            interval - number of element in each interval
            min_peak - minimum height of a peak
    return: peaks - list of all the local maxima
    '''

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
        if arr[i] < min_peak:
            mark[i] = False
        if mark[i]: 
            peaks.append(i)
    return peaks

def find_fund_freq(eigen_freq_lst, app_fund):
    ''' 
    purpose: find the fundamental frequency
    params: eigen_freq_lst - list of eigen frequencies and their amplitudes
            app_fund - approximate fundamental frequency
    return: fund_freq - fundamental frequency
    '''

    freq_lst   = []
    A_lst      = []
    for (freq, A) in eigen_freq_lst:
        freq_lst.append(freq)
        A_lst.append(A)
    # 求特征频率差分
    diff_freq = [freq_lst[0]]
    amplitude_weight = [A_lst[0]/2]
    for k in range(1, len(freq_lst)):
        diff_freq.append(freq_lst[k]-freq_lst[k-1])
        # 采用 min 函数，要求作差分的两个频率的幅值都较大
        # 这是一种较严格的要求，内涵上相当于逻辑与：&&
        amplitude_weight.append( min(A_lst[k], A_lst[k-1]) )
    # 投票算法，count 是字典，键为整数，值为小数票
    count = dict()
    # 乘子越大哈希值越发散，噪声点的投票越离散，从而越容易排除噪声干扰；
    # 但是过大会导致正确差分的投票也无法集中，从而无法得到正确的基频
    multiplier = 200
    # 定义哈希函数
    hash_func = lambda x: multiplier*x
    # 投票
    for j in range(len(diff_freq)):
        hash_val = hash_func(diff_freq[j])
        low_int  = int(hash_val)
        high_int = low_int + 1
        # 给附近的两个整数都投小数票，票数由相近度和振幅决定
        A_weight = amplitude_weight[j]
        count[low_int] =count.get(low_int, 0)+(high_int-hash_val)*A_weight
        count[high_int]=count.get(high_int,0)+(hash_val-low_int )*A_weight
    # 对字典依键进行从小到大的排序，键（哈希值）小的排前面
    count = dict(sorted(count.items()))
    # 统计得票最多的候选人，相同票数下优先选值小的候选人
    base_freq_map = 0
    ticket        = 0
    for (eigen, tic) in count.items():
        if tic > ticket:
            ticket = tic
            base_freq_map = eigen
    # 还原基频。采用以贡献票数为权的加权平均数
    fund_freq    = 0
    total_ticket = 0
    for j in range(len(diff_freq)):
        delta_val = abs( hash_func(diff_freq[j])-base_freq_map )
        if delta_val <= 1:
            # 权重 == 振幅权
            weight = amplitude_weight[j]
            fund_freq += weight * diff_freq[j]
            total_ticket += weight
    fund_freq /= total_ticket
    ratio = fund_freq / app_fund
    if (np.abs(ratio - int(ratio)) <= 0.05):
        fund_freq /= int(ratio)
    return fund_freq

def calc_derivative(arr, dt): 
    ''' 
    purpose: calculate the numerical derivative of the sequency arr
    params: arr - sequence
            dt - time num_sample (sampling period) 
    return: der - numerical derivative of the sequency arr
    formula: dx/dt = (x[t+dt] - x[t-dt]) / (2*dt) 
    '''

    der = np.zeros_like(arr)
    der[1 : -1] = (arr[2:] - arr[:-2]) / (2 * dt)
    der[0] = (arr[1] - arr[0]) / dt
    der[-1] = (arr[-1] - arr[-2]) / dt
    return der

def calc_modified_sig(acc, fs, fund_freq):
    ''' 
    purpose: calculate features of the modified signal
    params: acc - time history series of acceleration
            fs - sampling frequency
            fund_freq - fundamental frequency
    return: MPRS - PRS of modified signal
            MHCH - HCH of modified signal 
            AMHCH - approximate HCH of modified signal
    '''

    MPRS = 0
    MHCH = 0
    AMHCH = 0
    return MPRS, MHCH, AMHCH

def calc_VIV(acc, fs, app_fund=0.5):
    ''' 
    purpose: calculate features of a certain acceleration time history 
    params: acc - time history series of acceleration
            fs - sampling frequency
            app_fund - approximate fundamental frequency
    return: error_flag = 1 if accelerometer failures happen, else 0
            RMS - root mean square of acc
            PRS - peak ratio of spectrum
            HCH - hollow coefficient of Hilbert analytical signal of acc
            HCD - hollow coefficient of Derivative analytical signal of acc
            CCH - concentration coefficient of Hilbert analytical signal of acc
            CCD - concentration coefficient of Derivative analytical signal of acc
            MPRS - PRS of modified signal
            MHCH - HCH of modified signal 
            AMHCH - approximate HCH of modified signal
    '''

    Ts = 1 / fs
    np.nan_to_num(acc, copy=False, nan=9999)  # reset NaN as 9999
    num_sample = acc.shape[0]

    if np.abs(acc).max() > 9990 or np.abs(acc).max() < 1:
        # accelerometer failures happen, anyone == 9999 or all == 0
        return 1, 0, 0, 0, 0, 0, 0, 0, 0, 0

    # RMS
    RMS = np.sqrt(np.sum(acc ** 2) / num_sample)

    # PRS
    threshold = 0.8 * app_fund
    spec = fft(acc)
    f = fftfreq(acc.shape[0], Ts)  # f.max() == fs/2 : Nesquite's sampling theory
    half_f = f[:(acc.shape[0]+1) // 2]
    half_spec = np.abs(spec[:(acc.shape[0]+1) // 2])
    half_spec[0] /= 2
    wave_filter = (half_f > threshold)
    half_spec *= wave_filter
    half_spec /= half_spec.max()
    FSM_interval = int(1.5 * app_fund / half_f[1])
    peaks = np.array(find_peak(half_spec, FSM_interval, 0.))
    sorted_peaks = sorted(peaks, key=lambda i: half_spec[i])
    H1 = half_spec[sorted_peaks[-1]]
    H2 = half_spec[sorted_peaks[-2]]
    PRS = H2 / H1

    # HCH and CCH
    alpha = 0
    eps = 1e-12
    dt = Ts
    x = acc / np.abs(acc).max()
    y = hilbert(x)
    if np.abs(y).max() < eps:
        return 1, 0, 0, 0, 0, 0, 0, 0, 0, 0
    y /= np.abs(y).max()
    center_x = np.mean(x)
    center_y = np.mean(y)
    R = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    HCH = R.min() / R.max() 
    # CCH
    w = np.exp(- alpha * R)  # alpha = 0, 1, 2, 3...
    R_mean = np.sum(w * R) / np.sum(w)
    R_std = np.sqrt(np.sum(w * (R-R_mean)**2) / np.sum(w))
    CCH = R_mean / R_std
    
    # HCD and CCD
    y = calc_derivative(x, dt)
    if np.abs(y).max() < eps:
        # accelerometer failures happen, all == const
        return 1, 0, 0, 0, 0, 0, 0, 0, 0, 0
    y /= np.abs(y).max()
    center_x = np.mean(x)
    center_y = np.mean(y)
    R = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    HCD = R.min() / R.max()
    # CCD
    w = np.exp(- alpha * R)  # alpha = 0, 1, 2, 3...
    R_mean = np.sum(w * R) / np.sum(w)
    R_std = np.sqrt(np.sum(w * (R-R_mean)**2) / np.sum(w))
    CCD = R_mean / R_std

    # modified signal: MPRS, MHCH, AMHCH
    eigen_freq_lst = list(zip(half_f[peaks], half_spec[peaks]))
    fund_freq = find_fund_freq(eigen_freq_lst, app_fund)
    MPRS, MHCH, AMHCH = calc_modified_sig(acc, fs, fund_freq)

    # return
    return 0, RMS, PRS, HCH, HCD, CCH, CCD, MPRS, MHCH, AMHCH

def calc_VIV_s(acc, fs):
    ''' 
    purpose: simplified calc_VIV, merely calculate RMS and CCD
    params: acc - time history series of acceleration
            fs - sampling frequency
    return: error_flag = 1 if accelerometer failures happen, else 0
            RMS - root mean square of acc
            CCD - concentration coefficient of Derivative analytical signal of acc
    '''

    Ts = 1 / fs
    np.nan_to_num(acc, copy=False, nan=9999)  # reset NaN as 9999
    num_sample = acc.shape[0]

    if np.abs(acc).max() > 9990 or np.abs(acc).max() < 1:
        # accelerometer failures happen, anyone == 9999 or all == 0
        return 1, 0, 0, 0, 0, 0, 0, 0, 0, 0

    # RMS
    RMS = np.sqrt(np.sum(acc ** 2) / num_sample)

    # HCD and CCD
    alpha = 0.
    eps = 1e-12
    dt = Ts
    x = acc / np.abs(acc).max()
    y = calc_derivative(x, dt)
    if np.abs(y).max() < eps:
        # accelerometer failures happen, all == const
        return 1, 0, 0, 0, 0, 0, 0, 0, 0, 0
    y /= np.abs(y).max()
    center_x = np.mean(x)
    center_y = np.mean(y)
    R = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    HCD = R.min() / R.max()
    # CCD
    w = np.exp(- alpha * R)  # alpha = 0, 1, 2, 3...
    R_mean = np.sum(w * R) / np.sum(w)
    R_std = np.sqrt(np.sum(w * (R-R_mean)**2) / np.sum(w))
    CCD = R_mean / R_std

    # return
    return 0, RMS, CCD

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
            PRS - peak ratio of spectrum
            HCH - hollow coefficient of Hilbert analytical signal of acc
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
        return error_flag, 0, 0, 0, 0, 0, 0, 0, 0
    
    # RMS and HCD
    _, RMS, PRS, HCH, HCD = calc_VIV(acc, fs, app_fund_freq)

    # force 
    error_flag, fund_freq, max_A_freq, cable_force = \
            calc_force(acc, fs, app_fund_freq, m_bar, L, threshold=None)

    # maximum acc
    max_acc = acc.max()

    # return
    return error_flag, RMS, PRS, HCH, HCD, fund_freq, max_A_freq, max_acc, cable_force

def main():
    mat_file = "../res/algorithm/2021-04-23 10-VIC.mat"
    fs = 50
    acc = loadmat(mat_file)["data"].astype(np.float64)[:, 1]
    N = acc.shape[0] // 20
    a = acc[5*N: 6*N]  # 0*N: 1*N
    error_flag, RMS, PRS, HCH, HCD, CCH, CCD, PRSM, HCHM, AHCHM = calc_VIV(a, fs, app_fund=0.58)
    print(error_flag, RMS, PRS, HCH, HCD, CCH, CCD, PRSM, HCHM, AHCHM)
    error_flag, RMS, CCD = calc_VIV_s(a, fs)
    print(error_flag, RMS, CCD)

if __name__ == "__main__":
    main()
