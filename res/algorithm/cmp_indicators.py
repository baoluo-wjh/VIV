import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
plt.rc('font', family='Times New Roman')
plt.rc('text', usetex = True)

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
            markeredgecolor="tab:blue", markersize=4)
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
                            linewidth=4, color="tab:blue")
                    plt.text(xx, yy, f"{xx:.3f}", 
                            ha="center", va="bottom", fontsize=32, 
                            fontweight="bold", color="tab:blue")
                else:
                    # plt.plot([xx, xx], [0, yy], linestyle='--', 
                    #         linewidth=4, color="tab:blue", alpha=0.5)
                    ...
            ax.text(0.8, 0.6, 
                    f"{percentage*100:.0f}\%  of samples located in \n\
[{xs[0]:.3f}, {xs[-1]:.3f}]",
                    transform=ax.transAxes, ha="center", va="center", fontsize=48,
                    color="tab:green")
            plt.xlabel("$HCD - HCH$", size=64)
            plt.ylabel("PDF", size=64)
            plt.xticks(fontsize=48)
            plt.yticks(fontsize=48)
            plt.savefig(f"./distribution-HCH-HCD-{threshold:.1f}-{percentage:.2f}.jpg", 
                        bbox_inches="tight", pad_inches=0.05)
            plt.close()

            # scatter
            plt.figure(figsize=(20, 20))
            plt.axis("equal")
            plt.xlim([-0.01, 1.01])
            plt.ylim([-0.01, 1.01])
            plt.xticks(np.arange(0, 1.05, 0.1), fontsize=48)
            plt.yticks(np.arange(0, 1.05, 0.1), fontsize=48)
            plt.xlabel("HCH", size=64)
            plt.ylabel("HCD", size=64)
            # plt.plot(x, y, 'o', markerfacecolor = 'w', 
            #         markeredgecolor="tab:blue", markersize=4)
            plt.plot(inner[:, 0], inner[:, 1], '.', markerfacecolor = "tab:blue", 
                    markersize=1, alpha=0.5)    
            plt.plot(outer[:, 0], outer[:, 1], 'o', markerfacecolor = 'w', 
                    markeredgecolor="tab:blue", markersize=4)
            plt.plot([threshold, threshold], [threshold, 1.], linestyle='--', 
                    linewidth=4, color="tab:orange")
            plt.plot([threshold, 1.], [threshold, threshold], linestyle='--', 
                    linewidth=4, color="tab:orange")
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
line $HCD = HCH - {np.abs(xs[0]):.3f}$ \nand $HCD = HCH + {xs[-1]:.3f}$", 
                    ha="center", va="center", fontsize=48, color="tab:green")
            plt.text((threshold+1.)/2, threshold-0.01, f"Threshold = {threshold:.2f}",
                    ha="center", va="top", fontsize=48, color="tab:green")
            plt.savefig(f"./HCH-HCD-{threshold:.1f}-{percentage:.2f}.jpg", 
                        bbox_inches="tight", pad_inches=0.05)
            plt.close()

def cmp_CVH_CVD(CVH, CVD):
    x = CVH
    y = CVD
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
                            linewidth=4, color="tab:blue")
                    plt.text(xx, yy, f"{xx:.3f}", 
                            ha="center", va="bottom", fontsize=32, 
                            fontweight="bold", color="tab:blue")
                else:
                    # plt.plot([xx, xx], [0, yy], linestyle='--', 
                    #         linewidth=4, color="tab:blue", alpha=0.5)
                    ...
            ax.text(0.8, 0.6, 
                    f"{percentage*100:.0f}\%  of samples located in \n\
[{xs[0]:.3f}, {xs[-1]:.3f}]",
                    transform=ax.transAxes, ha="center", va="center", fontsize=48,
                    color="tab:green")
            plt.xlabel("$CVD - CVH$", size=64)
            plt.ylabel("PDF", size=64)
            plt.xticks(fontsize=48)
            plt.yticks(fontsize=48)
            plt.savefig(f"./distribution-CVH-CVD-{threshold:.1f}-{percentage:.2f}.jpg", 
                        bbox_inches="tight", pad_inches=0.05)
            plt.close()

            # scatter
            plt.figure(figsize=(20, 20))
            plt.axis("equal")
            plt.xlim([-0.01, 1.01])
            plt.ylim([-0.01, 1.01])
            plt.xticks(np.arange(0, 1.05, 0.1), fontsize=48)
            plt.yticks(np.arange(0, 1.05, 0.1), fontsize=48)
            plt.xlabel("CVH", size=64)
            plt.ylabel("CVD", size=64)
            # plt.plot(x, y, 'o', markerfacecolor = 'w', 
            #         markeredgecolor="tab:blue", markersize=4)
            plt.plot(inner[:, 0], inner[:, 1], '.', markerfacecolor = "tab:blue", 
                    markersize=1, alpha=0.5)    
            plt.plot(outer[:, 0], outer[:, 1], 'o', markerfacecolor = 'w', 
                    markeredgecolor="tab:blue", markersize=4)
            plt.plot([threshold, threshold], [threshold, 1.], linestyle='--', 
                    linewidth=4, color="tab:orange")
            plt.plot([threshold, 1.], [threshold, threshold], linestyle='--', 
                    linewidth=4, color="tab:orange")
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
line $CVD = CVH - {np.abs(xs[0]):.3f}$ \nand $CVD = CVH + {xs[-1]:.3f}$", 
                    ha="center", va="center", fontsize=48, color="tab:green")
            plt.text((threshold+1.)/2, threshold-0.01, f"Threshold = {threshold:.2f}",
                    ha="center", va="top", fontsize=48, color="tab:green")
            plt.savefig(f"./CVH-CVD-{threshold:.1f}-{percentage:.2f}.jpg", 
                        bbox_inches="tight", pad_inches=0.05)
            plt.close()

if __name__ == "__main__":
    csv_file = "../../software/data/ch02.csv"  # 02, 12, 20
    sample = pd.read_csv(csv_file).values
    # features = ["RMS", "PRS", "HCH", "HCD", "CVH", "CVD", "PRSM", "HCHM", "AHCHM"]
    right_index = (sample[:, 0] > 0)
    right_sample = sample[right_index]
    RMS = right_sample[:, 0]
    PRS = right_sample[:, 1]
    HCH = right_sample[:, 2]
    HCD = right_sample[:, 3]
    CVH = right_sample[:, 4]
    CVD = right_sample[:, 5]
    cmp_PRS_HCH(PRS, HCH)
    cmp_HCH_HCD(HCH, HCD)
    cmp_CVH_CVD(CVH, CVD)
