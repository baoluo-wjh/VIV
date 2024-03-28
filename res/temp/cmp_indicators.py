import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat
from h5py import File
from plot_single_sample import plot_time_spec_hil_deri
plt.rc('font', family='Times New Roman')
plt.rc('text', usetex = True)

import sys
sys.path.append("../../")
from software.train import plot_classified_sample
from software.calc import calc_VIV

def cmp_PRS_HCH(PRS, HCH):
    x = -1 + 2 / (PRS + 1)  # (1 - PRS) / (1 + PRS)
    y = HCH  # [0, 1]
    threshold_PRS = 0.1
    plt.figure(figsize=(20, 20))
    plt.axis("equal")
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xticks(np.arange(0, 1.05, 0.1), fontsize=48)
    plt.yticks(np.arange(0, 1.05, 0.1), fontsize=48)
    plt.xlabel("(1 - PRS) / (1 + PRS)", size=64)
    plt.ylabel("HCH", size=64)
    plt.plot(x, y, 'o', markerfacecolor = 'w', 
            markeredgecolor="tab:orange", markersize=4)
    plt.savefig("./PRS-HCH.jpg", bbox_inches="tight", pad_inches=0.05)

def cmp_HCH_HCD(HCH, HCD):
    x = HCH
    y = HCD
    for threshold in [0.3, 0.6]:
        for percentage in [0.95]:
            outer_flag: bool = (x >= threshold) & (y >= threshold)
            outer = np.c_[x[outer_flag], y[outer_flag]]
            inner = np.c_[x[outer_flag == False], y[outer_flag == False]]
            intercepts = outer[:, 1] - outer[:, 0]  # a = y - x
            N = intercepts.shape[0]
            avg = intercepts.mean()
            f = lambda shift: np.sum(((intercepts <= avg + shift) & \
                    (intercepts >= avg - shift)) / N) - percentage
            r = max(intercepts.max()-avg, avg-intercepts.min())
            l = 0
            while r - l >= 1e-6:
                m = (r + l) / 2
                if f(m) < 0:
                    l = m
                elif f(m) > 0:
                    r = m
                else:
                    break
            shift = (r + l) / 2

            # distribution
            _, ax = plt.subplots(figsize=(20, 10))
            n, bins, patches = plt.hist(intercepts, bins=100, color="tab:orange", 
                                        edgecolor='k', alpha=0.5, density=True)
            xs = [avg - shift, avg, avg + shift]
            ys = []
            for xx in xs:
                for r in range(1, len(bins)):
                    if bins[r] >= xx:
                        break
                l = r - 1
                ys.append(n[l])
            for i, (xx, yy) in enumerate(zip(xs, ys)):
                # yy = min(3 * yy, 0.75 * n.max())
                if i != 1:
                    yy = 0.2 * n.max()
                    plt.plot([xx, xx], [0, yy], linestyle='--', 
                            linewidth=4, color="tab:green")
                    plt.text(xx, yy, f"{xx:.3f}", 
                            ha="center", va="bottom", fontsize=32, 
                            fontweight="bold", color="tab:green")
                else:
                    # plt.plot([xx, xx], [0, yy], linestyle='--', 
                    #         linewidth=4, color="tab:blue", alpha=0.5)
                    ...
            ax.text(0.8, 0.38, 
                    f"{percentage*100:.0f}\%  of samples located \n\
within [{xs[0]:.3f}, {xs[-1]:.3f}]",
                    transform=ax.transAxes, ha="center", va="center", fontsize=48,
                    color="tab:green")
            plt.xlabel("$HCD - HCH$", size=64)
            plt.ylabel("PDF", size=64)
            plt.xticks(fontsize=48)
            plt.yticks(fontsize=48)
            plt.savefig(f"./HCH-HCD-{threshold:.1f}-{percentage:.2f}-dist.jpg", 
                        bbox_inches="tight", pad_inches=0.05)
            plt.close()

            # scatter
            plt.figure(figsize=(20, 20))
            plt.axis("equal")
            plt.xlim([-0.01, 1.01])
            plt.ylim([-0.01, 1.01])
            plt.xticks(np.arange(0, 1.05, 0.1), fontsize=48)
            plt.yticks(np.arange(0, 1.05, 0.1), fontsize=48)
            plt.xlabel("$x = HCH$", size=64)
            plt.ylabel("$y = HCD$", size=64)
            # plt.plot(x, y, 'o', markerfacecolor = 'w', 
            #         markeredgecolor="tab:blue", markersize=4)
            plt.plot(inner[:, 0], inner[:, 1], '.', c="tab:orange", alpha=0.25)       
            plt.plot(outer[:, 0], outer[:, 1], 'o', markerfacecolor = 'w', 
                    markeredgecolor="tab:orange", markersize=4)
            plt.plot([threshold, threshold], [threshold, 1.], linestyle='--', 
                    linewidth=4, color="tab:green")
            plt.plot([threshold, 1.], [threshold, threshold], linestyle='--', 
                    linewidth=4, color="tab:green")
            for i, intercept in enumerate(xs):
                if i == 1:
                    continue
                elif i == 0:
                    inner_x = np.linspace(-intercept, threshold-intercept, 
                                          100, endpoint=False)
                    outer_x = np.linspace(threshold-intercept, 1, endpoint=True)
                else:  # i == 2
                    inner_x = np.linspace(0, threshold, 100, endpoint=False)
                    outer_x = np.linspace(threshold, 1-intercept, endpoint=True)
                plt.plot(inner_x, inner_x+intercept, linestyle='--', 
                        linewidth=4, color="tab:green", alpha=0.5)
                plt.plot(outer_x, outer_x+intercept, linestyle='--', 
                        linewidth=4, color="tab:green")
            plt.text(0.7, 0.1, f"{percentage*100:.0f}\%  of samples located between \n\
$y = x - {np.abs(xs[0]):.3f}$ and $y = x + {xs[-1]:.3f}$", 
                    ha="center", va="center", fontsize=48, color="tab:green")
            plt.text((threshold+1.)/2, threshold+0.01, f"Threshold = {threshold:.2f}",
                    ha="center", va="bottom", fontsize=48, color="tab:blue")
            plt.savefig(f"./HCH-HCD-{threshold:.1f}-{percentage:.2f}.jpg", 
                        bbox_inches="tight", pad_inches=0.05)
            plt.close()

def cmp_CCH_CCD(CCH, CCD):
    x = CCH / CCH.max()
    y = CCD / CCD.max()
    for threshold in [0.3, 0.6]:
        for percentage in [0.95]:
            outer_flag: bool = (x >= threshold) & (y >= threshold)
            outer = np.c_[x[outer_flag], y[outer_flag]]
            inner = np.c_[x[outer_flag == False], y[outer_flag == False]]
            intercepts = outer[:, 1] - outer[:, 0]  # a = y - x
            N = intercepts.shape[0]
            avg = intercepts.mean()
            f = lambda shift: np.sum(((intercepts <= avg + shift) & \
                    (intercepts >= avg - shift)) / N) - percentage
            r = max(intercepts.max()-avg, avg-intercepts.min())
            l = 0
            while r - l >= 1e-6:
                m = (r + l) / 2
                if f(m) < 0:
                    l = m
                elif f(m) > 0:
                    r = m
                else:
                    break
            shift = (r + l) / 2

            # distribution
            _, ax = plt.subplots(figsize=(20, 10))
            n, bins, patches = plt.hist(intercepts, bins=100, color="tab:orange", 
                                        edgecolor='k', alpha=0.5, density=True)
            xs = [avg - shift, avg, avg + shift]
            ys = []
            for xx in xs:
                for r in range(1, len(bins)):
                    if bins[r] >= xx:
                        break
                l = r - 1
                ys.append(n[l])
            for i, (xx, yy) in enumerate(zip(xs, ys)):
                # yy = min(3 * yy, 0.75 * n.max())
                if i != 1:
                    yy = 0.2 * n.max()
                    plt.plot([xx, xx], [0, yy], linestyle='--', 
                            linewidth=4, color="tab:green")
                    plt.text(xx, yy, f"{xx:.3f}", 
                            ha="center", va="bottom", fontsize=32, 
                            fontweight="bold", color="tab:green")
                else:
                    # plt.plot([xx, xx], [0, yy], linestyle='--', 
                    #         linewidth=4, color="tab:blue", alpha=0.5)
                    ...
            ax.text(0.8, 0.38,
                    f"{percentage*100:.0f}\%  of samples located \n\
within [{xs[0]:.3f}, {xs[-1]:.3f}]", 
                    transform=ax.transAxes, ha="center", va="center", fontsize=48,
                    color="tab:green")
            ax.text(0.25, 0.6,
                    "$x = CCH / CCH_{max}$ \n $CCH_{max}=%.2f;$ \n\n" % CCH.max() +\
"$y = CCD / CCD_{max}$ \n $CCD_{max}=%.2f$" % CCD.max(),
                    transform=ax.transAxes, ha="center", va="center", fontsize=48, 
                    color="tab:blue")
            plt.xlabel("$y - x$", size=64)
            plt.ylabel("PDF", size=64)
            plt.xticks(fontsize=48)
            plt.yticks(fontsize=48)
            plt.savefig(f"./CCH-CCD-{threshold:.1f}-{percentage:.2f}-dist.jpg", 
                        bbox_inches="tight", pad_inches=0.05)
            plt.close()

            # scatter
            plt.figure(figsize=(20, 20))
            plt.axis("equal")
            plt.xlim([-0.01, 1.01])
            plt.ylim([-0.01, 1.01])
            plt.xticks(np.arange(0, 1.05, 0.1), fontsize=48)
            plt.yticks(np.arange(0, 1.05, 0.1), fontsize=48)
            plt.xlabel("$x = CCH / CCH_{max}$", size=64)
            plt.ylabel("$y = CCD / CCD_{max}$", size=64)
            plt.plot(inner[:, 0], inner[:, 1], '.', c="tab:orange", alpha=0.25)    
            plt.plot(outer[:, 0], outer[:, 1], 'o', markerfacecolor = 'w', 
                    markeredgecolor="tab:orange", markersize=4)
            plt.plot([threshold, threshold], [threshold, 1.], linestyle='--', 
                    linewidth=4, color="tab:blue")
            plt.plot([threshold, 1.], [threshold, threshold], linestyle='--', 
                    linewidth=4, color="tab:blue")
            for i, intercept in enumerate(xs):
                if i == 1:
                    continue
                elif i == 0:
                    inner_x = np.linspace(-intercept, threshold-intercept, 
                                          100, endpoint=False)
                    outer_x = np.linspace(threshold-intercept, 1, endpoint=True)
                else:  # i == 2
                    inner_x = np.linspace(0, threshold, 100, endpoint=False)
                    outer_x = np.linspace(threshold, 1-intercept, endpoint=True)
                plt.plot(inner_x, inner_x+intercept, linestyle='--', 
                        linewidth=4, color="tab:green", alpha=0.5)
                plt.plot(outer_x, outer_x+intercept, linestyle='--', 
                        linewidth=4, color="tab:green")
            plt.text(0.7, 0.1, 
                     f"{percentage*100:.0f}\%  of samples located between \n\
$y = x - {np.abs(xs[0]):.3f}$ and $y = x + {xs[-1]:.3f}$", 
                     ha="center", va="center", fontsize=48, color="tab:green")
            plt.text((threshold+1.)/2, threshold+0.01, f"Threshold = {threshold:.2f}",
                     ha="center", va="bottom", fontsize=48, color="tab:blue")
            plt.text((threshold+1.)/2, threshold-0.01, 
                     "$CCH_{max}=%.2f$\n$CCD_{max}=%.2f$" % (CCH.max(), CCD.max()),
                     ha="center", va="top", fontsize=48, color="tab:blue")
            plt.savefig(f"./CCH-CCD-{threshold:.1f}-{percentage:.2f}.jpg", 
                        bbox_inches="tight", pad_inches=0.05)
            plt.close()

def compare():
    csv_file = "../../software/data/ch02.csv"  # 02, 12, 20
    sample = pd.read_csv(csv_file).values
    # features = ["RMS", "PRS", "HCH", "HCD", "CCH", "CCD", "PRSM", "HCHM", "AHCHM"]
    right_index = (sample[:, 0] > 0)
    right_sample = sample[right_index]
    RMS = right_sample[:, 0]
    PRS = right_sample[:, 1]
    HCH = right_sample[:, 2]
    HCD = right_sample[:, 3]
    CCH = right_sample[:, 4]
    CCD = right_sample[:, 5]
    cmp_PRS_HCH(PRS, HCH)
    cmp_HCH_HCD(HCH, HCD)
    cmp_CCH_CCD(CCH, CCD)

def load_params(csv_dir, ch: str):
    with open(os.path.join(csv_dir, f"{ch}-params-train.json"), 'r') as f:
        data2save = json.load(f)

    # hyper-parameters
    train_percent   = data2save["train_percent"]
    extreme_percent = data2save["extreme_percent"]
    pre_sel_prob    = data2save["pre_sel_prob"]
    theta_eps       = data2save["theta_eps"]
    thresholds      = np.array(data2save["thresholds"])
    intercepts      = np.array(data2save["intercepts"])

    # trained results
    final_theta     = data2save["final_theta"]
    MF_pre_sel      = data2save["MF_pre_sel"]
    sample_sum      = data2save["sample_sum"]
    extreme_val     = data2save["extreme_val"]
    lines           = data2save["lines"]
    curves_1        = data2save["curves_1"]
    curves_2        = data2save["curves_2"]

    return train_percent, extreme_percent, pre_sel_prob, theta_eps, \
            thresholds, intercepts, final_theta, MF_pre_sel, sample_sum, \
            extreme_val, lines, curves_1, curves_2 

def identify(rms, ccd, curves_1, curves_2) -> str:
    x = rms
    y = ccd

    count_pro = 0
    for (a, b, threshold) in curves_1:
        if (x/a)**2+(y/b)**2 >= 1:
            count_pro += 1
    count_abs = 0
    for (a, b, intercept) in curves_2:
        if (x/a)**2+(y/b)**2 >= 1:
            count_abs += 1

    warning_level = "(%d, %d)" % (count_abs, count_pro)
    return warning_level

def find_all_mat(root):
    ''' 
    purpose: find all mat files in root 
    params: root - root path  
    return: mat_list - all mat files in root
    '''

    mat_list = []
    for file in os.listdir(root):
        file_path = os.path.join(root, file)
        if os.path.isdir(file_path):
            mat_list.extend(find_all_mat(file_path))
        else:
            file_name, ext = file.rsplit('.', 1)
            if ext != "mat":
                continue
            time, kind = file_name.rsplit('-', 1)
            if kind == "VIC":
                mat_list.append(file_path)
    return mat_list

def get_mat_minute(mat_index_table, mat_list, index):
    for (i, r) in enumerate(mat_index_table):
        if index < r:
            return (mat_list[i], index - (mat_index_table[i - 1] if i >= 1 else 0))
        
def plot_special(ch, fts, df=None, special_index=None, cut_rate=0.):
    csv_dir = "../../software/data"
    minute_interval = 1
    fs = 50
    n = int(minute_interval * 60 * fs)

    sample = pd.read_csv(os.path.join(csv_dir, f"{ch}.csv")).values
    index = pd.read_csv(os.path.join(csv_dir, f"{ch}-index.csv"), 
                        header=None).values.flatten()
    X = sample[sample[:, 0] > 0][:, fts]

    if df:
        for i, (x, y) in enumerate(X):
            if df(x, y):
                print(i, index[i], X[i], sample[index[i], fts])

    # train_percent, extreme_percent, pre_sel_prob, theta_eps, \
    #         thresholds, intercepts, final_theta, MF_pre_sel, sample_sum, \
    #         extreme_val, lines, curves_1, curves_2 = load_params(csv_dir, ch)
    # rms, ccd = X[234445]
    # print(identify(rms, ccd, curves_1, curves_2))
    # plot_classified_sample(
    #     np.array([[rms, ccd]]),
    #     "./ch02-loc2.jpg",
    #     None, 
    #     extreme_val,
    #     False, 
    #     ["RMS [gal]", "CCD"], 
    #     curves=[curves_1, curves_2]
    # )

    if special_index:
        mat_list = sorted(find_all_mat("../../../raw-data/Tongling"))
        mat_index_table = pd.read_csv("../../software/mat_index_table.csv", 
                                      header=None).values.flatten()
        mat_file, minute = get_mat_minute(mat_index_table, mat_list, special_index)
        print(mat_file, minute)
        try:
            acc = loadmat(mat_file)["data"].astype(np.float64)
        except:
            acc = np.transpose(File(mat_file, 'r')["data"]).astype(np.float64)
        a = acc[minute*n: (minute+1)*n, int(ch[-2:]) - 1]
        error_flag, RMS, PRS, HCH, HCD, CCH, CCD, PRSM, HCHM, AHCHM = \
                calc_VIV(a, fs, 0.58, cut_rate)
        print(error_flag, RMS, PRS, HCH, HCD, CCH, CCD, PRSM, HCHM, AHCHM)
        plot_time_spec_hil_deri(
            a, 
            f"{ch}-{mat_file.rsplit('/', 1)[-1].rsplit('-', 1)[0]}-{minute}-{minute+1}.jpg",
            cut_rate
        )

def plot_special_ch02_ch20():
    plot_special("ch02", [0, 5], 
                 lambda x, y: x <= 100 and y >= 30, 
                 special_index=None)
    plot_special("ch02", [0, 5], 
                 None, 
                 special_index=252089)
    plot_special("ch20", [0, 5], 
                 lambda x, y: x <= 40 and y >= 20, 
                 special_index=None)
    plot_special("ch20", [0, 5], 
                 None, 
                 special_index=239048)

def plot_HC_vs_CC():
    # features = ["RMS", "PRS", "HCH", "HCD", "CCH", "CCD", "PRSM", "HCHM", "AHCHM"]

    # # CCH vs CCD
    # plot_special("ch02", [4, 5], 
    #              lambda x, y: x <= 0.5 * 39.65 and y >= 0.7 * 40.96, 
    #              special_index=None)
    # plot_special("ch02", [4, 5], 
    #              None, 
    #              special_index=123852,
    #              cut_rate=0.)

    # # HCH vs HCD
    # plot_special("ch02", [2, 3], 
    #              lambda x, y: x <= 0.1 and y >= 0.73,
    #              special_index=None)
    plot_special("ch02", [2, 3], 
                 None,
                 special_index=134130,
                 cut_rate=0.)

if __name__ == "__main__":
    compare()
