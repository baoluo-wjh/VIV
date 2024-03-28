import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from time import time
from scipy.optimize import minimize, brentq, fsolve
from concurrent.futures import ProcessPoolExecutor
# from sklearn.ensemble import IsolationForest
# from sklearn.neighbors import LocalOutlierFactor
# from sklearn.svm import OneClassSVM
# from sklearn.linear_model import SGDOneClassSVM

plt.rc('font', family='Times New Roman', size=48)  # 24
plt.rc('text', usetex = True)

def get_sample(csv_file):
    ''' 
    purpose: get all the right samples (without 9999)
    params: csv_file - csv file that records all the (RMS, PRS, ...)
    return: right_sample - shape == (length x 2)
    '''

    sample = pd.read_csv(csv_file).values
    right_index = (sample[:, 0] > 0)  # RMS > 0
    # features = ["RMS", "PRS", "HCH", "HCD", "CCH", "CCD", "MPRS", "MHCH", "AMHCH"]
    right_sample = sample[right_index][:, [0, 5]]  # RMS and CCD
    return right_sample

def get_modified_feature(x, trans="norm", eps_t=6, degree=2):
    ''' 
    purpose: get modified feature
    params: x - original feature
            trans - trans method
            eps_t - eps=0.1**eps_t
            degree - power
    return: x_modi - modified feature
    '''

    if trans == "norm":
        x_modi = x / x.max()
    elif trans == "log":
        eps = 0.1 ** eps_t
        x_modi = -np.log10(x + eps)
        x_modi[x_modi < 0] = 0
        x_modi /= eps_t
        x_modi = 1 - x_modi
    elif trans == "degree":
        x_modi = x ** degree

    return x_modi

def get_mixed_feature(X, main_direction):
    ''' 
    purpose: get mixed feature
    params: X - sample
            main_direction - main direction
    return: MF - mixed feature
    '''

    alpha, beta = main_direction
    MF = X[:, 0] * alpha + X[:, 1] * beta
    return MF

def calc_quantile(x, threshold, eps=1e-5):
    ''' 
    purpose: calc quantile
    params: x - feature
            threshold - threshold percentage
            eps - critical distance between left and right
    return: m - middle value
    '''

    f = lambda x0: np.sum(x <= x0) / x.shape[0] - threshold
    l = x.min()
    r = x.max()
    # dichotomy
    while r - l >= eps:
        m = (r + l) / 2
        if f(m) < 0:
            l = m
        elif f(m) > 0:
            r = m
        else:
            break
    return (r + l) / 2

def plot_distribution(x, prefix, f_name="MF", 
                      x_range=None, hist_bins=250, threshold=None):
    ''' 
    purpose: plot distribution
    params: x - feature
            prefix - path and channel
            f_name - feature name
            x_range - xlim
            hist_bins - hist bins
            threshold - threshold percentage
    return: None
    '''

    # plot pdf
    _, ax = plt.subplots(figsize=(20, 10))
    n, bins, patches = plt.hist(x, bins=hist_bins, color="tab:orange", 
                                edgecolor='k', alpha=0.5, density=True)

    if x_range:
        plt.xlim(x_range)
    if threshold:
        K = calc_quantile(x, threshold)
        for r in range(1, len(bins)):
            if bins[r] >= K:
                break
        l = r - 1
        height = min(3*n[l], 0.75*n.max())
        plt.plot([K, K], [0, height], linestyle='-', 
                 linewidth=4, color="tab:blue")
        plt.text(K, height, 
                 f"{K:.3f}: Threshold for\n{100*threshold:.1f}\% Probability", 
                 ha="center", va="bottom", fontsize=32, 
                 fontweight="bold", color="tab:blue")
    plt.xlabel(f_name, size=64)
    plt.ylabel("PDF", size=64)
    plt.xticks(fontsize=48)
    plt.yticks(fontsize=48)
    plt.savefig(f"{prefix}-{f_name}-PDF.jpg", 
                bbox_inches="tight", pad_inches=0.05)
    plt.close()

    # # plot cdf
    # _, ax = plt.subplots(figsize=(20, 10))
    # plt.hist(x, bins=hist_bins, color="tab:orange", edgecolor='k', alpha=0.5, 
    #          density=True, cumulative=True)
    # if x_range:
    #     plt.xlim(x_range)
    # if threshold:
    #     K = calc_quantile(x, threshold)
    #     plt.plot([0, K], [threshold, threshold], linestyle='--', 
    #              linewidth=4, color="tab:blue", alpha=0.5)
    #     plt.plot([K, K], [threshold, 0], linestyle='--', 
    #              linewidth=4, color="tab:blue", alpha=0.5)
    #     plt.scatter([K], [threshold], s=512, c="tab:blue", marker="*")
    #     plt.text(K, threshold, f"({K:.3f}, {threshold:.3f})", ha="center", 
    #              va="bottom", fontsize=32, fontweight="bold", color="tab:blue")
    # plt.xlabel(f_name, size=64)
    # plt.ylabel("CDF", size=64)
    # plt.xticks(fontsize=48)
    # plt.yticks(fontsize=48)
    # plt.savefig(f"{prefix}-{f_name}-CDF.jpg", 
    #             bbox_inches="tight", pad_inches=0.05)
    # plt.close()

def get_sample_sum(X):
    ''' 
    purpose: get sample sum
    params: X - sample
    return: A - [a3, a2, a1, a0], a3 = sum(s*s*s), a2 = sum(s*s*t)...
            B - [b2, b1, b0], b2 = sum(s*2), b1 = sum(s*t), b0 = sum(t*t)
            C - [c1, c0], c1 = sum(s), c0 = sum(t)
            N - number of used samples
    '''

    s = X[:, 0]
    t = X[:, 1]
    a3 = np.sum(s*s*s)
    a2 = np.sum(s*s*t)
    a1 = np.sum(s*t*t)
    a0 = np.sum(t*t*t)
    b2 = np.sum(s*s)
    b1 = np.sum(s*t)
    b0 = np.sum(t*t)
    c1 = np.sum(s)
    c0 = np.sum(t)
    A = [a3, a2, a1, a0]
    B = [b2, b1, b0]
    C = [c1, c0]
    N = s.shape[0]
    return A, B, C, N

def get_dot_algo(a3, a2, a1, a0, b2, b1, b0, c1, c0, n, thr, theta):
    ''' 
    purpose: input parameters to test_dot
    params: A - [a3, a2, a1, a0], a3 = sum(s*s*s), a2 = sum(s*s*t)...
            B - [b2, b1, b0], b2 = sum(s*2), b1 = sum(s*t), b0 = sum(t*t)
            C - [c1, c0], c1 = sum(s), c0 = sum(t)
            n - number of samples
            thr - threshold
            theta - weighted-PCA theta
    return: test_dot - test function
    '''

    # direction vector
    lambd = np.cos(theta)
    mu = np.sin(theta)

    # centroid
    sc = (lambd*b2+mu*b1-thr*c1) / (lambd*c1+mu*c0-thr*n)
    tc = (lambd*b1+mu*b0-thr*c0) / (lambd*c1+mu*c0-thr*n)

    # sigma
    # Always use thr*n rather than thr! I debugged for three days!
    sigma_12 = (lambd*a2+mu*a1-thr*b1) - (lambd*b2+mu*b1-thr*c1)*tc - \
            (lambd*b1+mu*b0-thr*c0)*sc + (lambd*c1+mu*c0-thr*n)*sc*tc 
    sigma_11 = (lambd*a3+mu*a2-thr*b2) - 2*(lambd*b2+mu*b1-thr*c1)*sc + \
            (lambd*c1+mu*c0-thr*n)*sc*sc 
    sigma_22 = (lambd*a1+mu*a0-thr*b0) - 2*(lambd*b1+mu*b0-thr*c0)*tc + \
            (lambd*c1+mu*c0-thr*n)*tc*tc 
    # multiplier = (lambd * c1 + mu * c0 - thr * n) / (
    #     lambd ** 2 * (c1 ** 2 - b2) +\
    #     mu ** 2 * (c0 ** 2 - b0) +\
    #     thr ** 2 * (n ** 2 - n) +\
    #     2 * lambd * mu * (c1 * c0 - b1) -\
    #     2 * mu * thr * c0 * (n - 1) -\
    #     2 * thr * lambd * c1 * (n - 1)
    # )
    # sigma_11 *= multiplier
    # sigma_12 *= multiplier
    # sigma_22 *= multiplier

    # orthogonal
    b = np.array([sigma_11*lambd+sigma_12*mu, sigma_12*lambd+sigma_22*mu])
    b /= np.sqrt((b**2).sum())
    cos_phi = -mu*b[0] + lambd*b[1]
    return cos_phi

def get_dot_trad(X_sqr, theta, MF_pre_sel=None, pre_sel_prob=0.95):
    ''' 
    purpose: get inner product using the traditional method
    params: X_sqr - squared data
            theta - extreme percent
            MF_pre_sel
            pre_sel_prob - 
    return: cos < l, Sigma @ k >
    '''

    # data
    s = X_sqr[:, 0]
    t = X_sqr[:, 1]
    lambd = np.cos(theta)
    mu = np.sin(theta)

    # weight
    MF = lambd * s + mu * t
    if MF_pre_sel:
        w = MF - MF_pre_sel
    else:
        w = np.maximum(0, MF - calc_quantile(MF, pre_sel_prob))

    # inner product
    sc = np.sum(w * s) / np.sum(w)
    tc = np.sum(w * t) / np.sum(w)
    u = s - sc
    v = t - tc
    sigma_11 = np.sum(w * u * u)
    sigma_12 = np.sum(w * u * v)
    sigma_22 = np.sum(w * v * v)
    # multiplier = np.sum(w) / (np.sum(w) ** 2 - np.sum(w ** 2))
    # sigma_11 *= multiplier
    # sigma_12 *= multiplier
    # sigma_22 *= multiplier
    b = np.array([sigma_11*lambd+sigma_12*mu, sigma_12*lambd+sigma_22*mu])
    b /= np.sqrt((b**2).sum())
    val = -mu*b[0] + lambd*b[1]

    return val

def brentq_minimize(test_dot, theta_eps):
    ''' 
    purpose: get weighted-PCA theta, use brentq and minimize
    params: test_dot - function pointer: input theta, output cos
            theta_eps - tolerance for two consecutive theta values
    return: theta_opt - optimized theta for weighted-PCA 
    '''

    # create intervals
    K = 100
    eps = theta_eps
    inter_len = np.zeros(K,)
    n = np.arange(K // 2)
    B = 6 * (np.pi - 2 * K * eps) / ((K - 2) * (K - 1) * K)
    inter_len[:K // 2] = eps + B * n ** 2
    inter_len[K // 2:] = inter_len[K // 2 - 1: : -1]
    end_p = inter_len.cumsum()

    # brentq method
    former_r = 0
    former_f = test_dot(former_r)
    for i in range(K):
        r = end_p[i]
        f = test_dot(r)
        if former_f * f <= 0:
            l = former_r
            theta_opt = brentq(test_dot, l, r, xtol=theta_eps)
            return theta_opt
        former_r = r
        former_f = f
    
    # minimize
    former_r = 0
    min_f = 6.22
    min_x = -6.22
    for i in range(K):
        l = former_r
        r = end_p[i]
        res = minimize(lambda x: np.abs(test_dot(x)), 
                       x0=l, bounds=[(l, r)], tol=theta_eps)
        x = res.x[0]
        f = res.fun
        if f < min_f:
            min_f = f
            min_x = x
        former_r = r
    return min_x
    
def fsolve_brentq_minimize(test_dot, pre_theta, theta_eps):
    ''' 
    purpose: get weighted-PCA theta, use fsolve, brentq, and minimize
    params: test_dot - function pointer: input theta, output cos
            pre_theta - initial theta for fsolve
            theta_eps - tolerance for two consecutive theta values
    return: theta_opt - optimized theta for weighted-PCA 
    '''

    # method 1: fsolve
    theta_opt = fsolve(test_dot, x0=pre_theta, xtol=theta_eps)[0]
    theta_opt = theta_opt - np.floor(theta_opt/(2*np.pi))*2*np.pi
    if np.abs(test_dot(theta_opt)) <= theta_eps and theta_opt <= np.pi / 2:
        return theta_opt

    # method 2 and 3: brentq and minimize
    return brentq_minimize(test_dot, theta_eps)

def iter_theta(csv_dir, ch_name, X_sqr, pre_sel_prob, 
               theta_eps, init_pre_theta=0.):
    ''' 
    purpose: iteration to find the final theta (pre_theta = theta_opt)
    params: csv_dir - csv directory
            ch_name - channel name
            X_sqr - squared data
            pre_sel_prob - probability threshold for pre-selection
            theta_eps - tolerance for two consecutive theta values
            init_pre_theta - initial theta for sample pre-selection 
    return: final_theta - final theta
            MF_pre_sel - MF value for sample pre-selection
            A, B, C, N - sample sums
    '''

    theta_opt_lst = [init_pre_theta]
    count_iter = 1

    while True:
        # pre-selection
        pre_theta = theta_opt_lst[-1]
        MF = get_mixed_feature(X_sqr, (np.cos(pre_theta), np.sin(pre_theta)))
        MF_pre_sel = calc_quantile(MF, pre_sel_prob)
        ign_info = (MF < MF_pre_sel)
        A, B, C, N = get_sample_sum(X_sqr[~ign_info])

        # find the optimal theta
        test_dot = lambda t: get_dot_algo(*A, *B, *C, N, MF_pre_sel, t)
        theta_opt = fsolve_brentq_minimize(test_dot, pre_theta, theta_eps)
        # print(count_iter, pre_theta, theta_opt, test_dot(theta_opt))

        # plot theta_vs_cos diagram
        if csv_dir and ch_name:
            pic_path = os.path.join(csv_dir, "%s-%d-pre_theta=%.4fpi.jpg" \
                                    % (ch_name, count_iter, pre_theta/np.pi))
            plot_theta_cos(test_dot, theta_opt, pic_path)

        # compare
        if np.abs(pre_theta - theta_opt) < theta_eps and \
                np.abs(test_dot(theta_opt)) < theta_eps:
            break

        # append current theta_opt to theta_opt_lst
        if count_iter <= 5:
            theta_opt_lst.append(theta_opt)
        elif count_iter <= 15:  
            # double fixed points, very rare, every important!
            theta_opt_lst.append( (theta_opt + pre_theta) / 2 )
        elif count_iter < 100:
            # multiple fixed points, very rare, every important!
            theta_opt_lst.append((sum(theta_opt_lst[-10:]) + theta_opt) / 11)
        else:
            # cannot solve, so use get_dot_trad
            theta = brentq_minimize(
                lambda t: get_dot_trad(X_sqr, t, None, pre_sel_prob), 
                theta_eps
            )
            MF = get_mixed_feature(X_sqr, (np.cos(theta), np.sin(theta)))
            MF_pre_sel = calc_quantile(MF, pre_sel_prob)
            ign_info = (MF < MF_pre_sel)
            A, B, C, N = get_sample_sum(X_sqr[~ign_info])
            return theta, MF_pre_sel, A, B, C, N

        # next
        count_iter += 1

    return theta_opt_lst[-1], MF_pre_sel, A, B, C, N

def plot_theta_cos(test_dot, theta_opt, pic_path):
    ''' 
    purpose: plot all theta within [0, pi/2]
    params: test_dot - function pointer: input theta, output cos
            theta_opt - optimized theta
            pic_path - picture path
    return: None
    '''

    theta = np.linspace(0, np.pi/2, 1001, endpoint=True)
    dot_prod = np.array([test_dot(t) for t in theta])
    _, ax = plt.subplots(figsize=(20, 10))
    plt.plot(theta, dot_prod, color="tab:blue", linewidth=4)
    plt.xlim(0, np.pi/2)
    plt.ylim(dot_prod.min() - 0.03, dot_prod.max() + 0.03)
    plt.xlabel("$\\theta$ [rad]", size=64)
    plt.ylabel("$COS < l, \\Sigma k >$", size=64)
    plt.xticks(np.pi/12 * np.arange(7), 
               ["0"] + ["$\\frac{%d}{12} \\pi$" % num for num in np.arange(1, 7)], 
               fontsize=48)
    plt.yticks(fontsize=48)
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))
    x0 = theta_opt
    y0 = test_dot(x0)
    plt.scatter([x0], [y0], s=512, c="tab:orange", marker="*")
    if x0 >= np.pi / 4:
        plt.text(x0 - 0.03, y0, "$%.4f \\; (%.4f \\pi)$" % (x0, x0 / np.pi), 
                 ha="right", va="center", fontsize=64, color="tab:green")
    else:
        plt.text(x0 + 0.03, y0, "$%.4f \\; (%.4f \\pi)$" % (x0, x0 / np.pi), 
                 ha="left", va="center", fontsize=64, color="tab:green")
    plt.savefig(pic_path, bbox_inches="tight", pad_inches=0.05)
    plt.close()

def plot_theta_cos_trad_vs_algo(X_sqr, pre_sel_prob, test_dot, theta_opt, pic_path):
    ''' 
    purpose: plot all theta within [0, pi/2]
    params: X_sqr - squared data
            pre_sel_prob - pre-selection probability
            test_dot - function pointer: input theta, output cos
            theta_opt - optimized theta
            pic_path - picture path
    return: None
    '''

    # selected data
    MF = get_mixed_feature(X_sqr, [np.cos(theta_opt), np.sin(theta_opt)])
    MF_pre_sel = calc_quantile(MF, pre_sel_prob)
    ign_info = (MF < MF_pre_sel)

    T = np.linspace(0, np.pi/2, 1001, endpoint=True)
    dot_0 = np.array([test_dot(t) for t in T])
    dot_1 = np.array([get_dot_trad(X_sqr[~ign_info], t, MF_pre_sel) for t in T])
    dot_2 = np.array([get_dot_trad(X_sqr, t, None, pre_sel_prob) for t in T])
    dot_mat = np.c_[dot_0, dot_1, dot_2]

    _, ax = plt.subplots(figsize=(20, 10))
    leg = []
    for i, (c, ls) in enumerate(zip(["tab:blue", "tab:red", "tab:cyan"], 
                                    ["-", ":", "--"])):
        leg.append(plt.plot(T, dot_mat[:, i], color=c, 
                            linestyle=ls, linewidth=4)[0])
    plt.rc('font', size=24)
    plt.legend(
        leg, 
        ["Proposed", "Trad (no ReLU)", "Trad (with ReLU)"],
        fontsize=32, 
        loc="best", 
        title="Same Best $\\theta$ in Different Methods"
    )
    plt.xlim(0, np.pi/2)
    plt.ylim(dot_mat.min() - 0.03, dot_mat.max() + 0.03)
    plt.xlabel("$\\theta$ [rad]", size=64)
    plt.ylabel("$COS < l, \\Sigma k >$", size=64)
    plt.xticks(np.pi/12 * np.arange(7), 
            ["0"] + ["$\\frac{%d}{12} \\pi$" % num for num in np.arange(1, 7)], 
            fontsize=48)
    plt.yticks(fontsize=48)
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))
    x0 = theta_opt
    y0 = test_dot(x0)
    plt.scatter([x0], [y0], s=512, c="tab:orange", marker="*")
    if x0 >= np.pi / 4:
        plt.text(x0 - 0.03, y0, "$%.4f \\; (%.4f \\pi)$" % (x0, x0 / np.pi), 
                 ha="right", va="center", fontsize=64, color="tab:green")
    else:
        plt.text(x0 + 0.03, y0, "$%.4f \\; (%.4f \\pi)$" % (x0, x0 / np.pi), 
                 ha="left", va="center", fontsize=64, color="tab:green")
    plt.savefig(pic_path, bbox_inches="tight", pad_inches=0.05)
    plt.close()

def plot_classified_sample(X, pic_path, 
                           ign_info, extreme_val, 
                           is_normed, labels, 
                           lines=None, curves=None):
    ''' 
    purpose: plot lined samples 
    params: X - sample, None or (1, 2) or (N, 2)
            pic_path - picture path
            ign_info - ignored indexes
            extreme_val - extreme values
            is_normed - whether x is normed to [0, 1]
            labels - x label and y label
            lines - decision lines in the square_y-square_x plain
            curves - decision curves in the norm_y-raw_x plain
    return: None
    '''

    if is_normed:
        plt.figure(figsize=(20, 20))
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xticks([0, 0.5, 1], fontsize=48)
        plt.yticks([0, 0.5, 1], fontsize=48)
        plt.axis("equal")   
    else:
        plt.figure(figsize=(20, 10))
        plt.xticks(fontsize=48)
        plt.yticks(fontsize=48)

    plt.xlabel(labels[0], size=64)
    plt.ylabel(labels[1], size=64)

    sample_num = X.shape[0]
    if sample_num > 1:
        ext_info = (X[:, 0] >= extreme_val[0] * (1 - 1e-9)) | \
                   (X[:, 1] >= extreme_val[1] * (1 - 1e-9))
        X_ign = X[ign_info]               # ignored points
        X_ext = X[~ign_info & ext_info]   # reference but no extreme points
        X_ref = X[~ign_info & ~ext_info]  # extreme points
        fr_1 = plt.plot(X_ext[:, 0], X_ext[:, 1], '*', 
                        markerfacecolor="tab:red", 
                        markeredgecolor="tab:red", markersize=4)
        fr_2 = plt.plot(X_ref[:, 0], X_ref[:, 1], 'o', 
                        markerfacecolor='w', 
                        markeredgecolor="tab:orange", markersize=4)
        fr_3 = plt.plot(X_ign[:, 0], X_ign[:, 1], 'o', 
                        markerfacecolor="tab:green", 
                        markeredgecolor="tab:green", markersize=1, alpha=0.5)
    elif sample_num == 1:
        plt.plot(X[:, 0], X[:, 1], '*', markerfacecolor="tab:orange", 
                markeredgecolor="tab:orange", markersize=32)

    if lines:
        leg_1 = []
        thresholds = []
        num_line = len(lines)
        count = 0
        for (k, b, threshold) in lines:
            count += 1
            color_used = '#00'+str(hex(int(0xff*(count/num_line))))[2:]+'ff'
            # x = np.linspace(0, 1, 10001, endpoint=True)
            # y = k * x + b
            # inner_index = ((y >= 0) & (y <= 1))
            # xx = x[inner_index]
            # yy = y[inner_index]
            a = -b / k
            if a <= 1:
                if b <= 1:
                    xx = np.array([0, a])
                    yy = np.array([b, 0])
                else:
                    xx = np.array([(1-b)/k, a])
                    yy = np.array([1, 0])
            else:
                if b <= 1:
                    xx = np.array([0, 1])
                    yy = np.array([b, k+b])
                elif k+b <= 1: 
                    xx = np.array([(1-b)/k, 1])
                    yy = np.array([1, k+b])
                else:
                    xx = np.array([])
                    yy = np.array([])
            fr = plt.plot(xx, yy, color=color_used, linestyle="--", linewidth=4)
            leg_1.append(fr[0])
            thresholds.append(threshold)
            if count == 1:
                if sample_num > 1:
                    X_ign = X[ign_info]
                    x = X_ign[:, 0]
                    y = X_ign[:, 1]
                    x_ky = x + k * y
                    c = np.array([x_ky.max(), x_ky.min()])
                    x_ = (c - k * b) / (k ** 2 + 1)
                    y_ = (k * c + b) / (k ** 2 + 1)
                    delta_x = x_[0] - x_[1]
                    delta_y = y_[1] - y_[0]
                    x0 = x_.mean()
                    y0 = y_.mean()
                else:  # sample_num == 1
                    a = -b / k
                    delta_x = a
                    delta_y = b
                    x0 = a / 2
                    y0 = b / 2
                length = np.sqrt(delta_x ** 2 + delta_y ** 2)
                cos_ = delta_y / length
                sin_ = delta_x / length
                L = length / 4
                W = 0.1 * L 
                key_points = np.array([
                    [0, -W/2],
                    [L, -W/2],
                    [L, -W],
                    [3/2*L, 0],
                    [L, W],
                    [L, W/2],
                    [0, W/2],
                ])
                key_points = key_points @ np.array([
                    [cos_, sin_], 
                    [-sin_, cos_]
                ]) + np.array([x0, y0])
                arrow = mpatches.Polygon(
                    key_points, 
                    color=color_used, 
                    linewidth=2, fill=False, 
                    # larger the zorder is, closer it is to eyes
                    # default -- patch: 1, line, point: 2, text: 3, legend: 5
                    zorder=5)
                plt.gca().add_patch(arrow)
        plt.rc('font', size=36)
        legend_1 = plt.legend(leg_1, ["%.1f" % (100*threshold) for threshold in thresholds], 
                              fontsize=48, loc="upper right", 
                              title="Probability [\%]")
        if sample_num > 1:
            legend_0 = plt.legend([fr_1[0], fr_2[0], fr_3[0]], 
                                  ["Extreme", "Referred", "Ignored"], 
                                  fontsize=48, loc="upper left", 
                                  title="Sample Points")
            plt.gca().add_artist(legend_1)

    if curves:
        leg_1 = []
        leg_2 = []
        thresholds = []
        intercepts = []
        num_curve = len(curves[0]) + len(curves[1])
        count = 0
        for (i, curves_i) in enumerate(curves):
            for (a, b, mark) in curves_i:
                count += 1
                color_used ='#00'+str(hex(int(0xff*(count/num_curve))))[2:]+'ff'
                # x = np.linspace(0, a, 10001, endpoint=True)
                # y = b * np.sqrt(1 - (x/a) ** 2)
                # if is_normed:
                #     inner_index = (x <= 1) & (y <= 1)
                # else:
                #     inner_index = y <= np.maximum(extreme_val[1], X[:, 1].max())
                # xx = x[inner_index]
                # yy = y[inner_index]
                f = lambda x: b * np.sqrt(1 - (x/a) ** 2)
                g = lambda y: a * np.sqrt(1 - (y/b) ** 2)
                if is_normed:
                    if a <= 1:
                        if b <= 1:
                            xx = np.linspace(0, a, 101, endpoint=True)
                        else:
                            xx = np.linspace(g(1), a, 101, endpoint=True)
                    else:
                        if b <= 1:
                            xx = np.linspace(0, 1, 101, endpoint=True)
                        elif f(1) <= 1:
                            xx = np.linspace(g(1), 1, 101, endpoint=True)
                        else:
                            xx = np.array([])
                else:
                    max_y = np.maximum(extreme_val[1], X[:, 1].max())
                    if b <= max_y:
                        xx = np.linspace(0, a, 101, endpoint=True)
                    else:
                        xx = np.linspace(g(max_y), a, 101, endpoint=True)
                yy = f(xx)
                fr = plt.plot(
                    xx, yy, 
                    color=color_used, 
                    linewidth=4, linestyle="--" if i == 0 else ':'
                )
                if i == 0:
                    leg_1.append(fr[0])
                    thresholds.append(mark)
                else:  # boundary
                    leg_2.append(fr[0])
                    intercepts.append(mark)
                    if is_normed and (mark <= 1 or b*np.sqrt(1-(1/a)**2) <= 1):
                        plt.plot([mark], [0.], '*', markersize=32, 
                                 markerfacecolor='w', markeredgecolor=color_used)
                        plt.text(mark, -0.02, "%.3f" % mark, ha="center", 
                                 va="top", fontsize=32, color=color_used)
                    elif not is_normed:
                        plt.plot([mark], [0.], '*', markersize=32, 
                                 markerfacecolor='w', markeredgecolor=color_used)
                        plt.text(mark + 5, 0., "(%.0f, 0)" % mark, ha="left", 
                                 va="center", fontsize=32, color=color_used)
        fs = 48 if is_normed else 32
        plt.rc('font', size=fs//4*3) 
        legend_1 = plt.legend(
            leg_1, 
            ["%.1f" % (100*threshold) for threshold in thresholds], 
            loc="upper right", fontsize=fs, title="Probability [\%]"
        )
        legend_2 = plt.legend(
            leg_2, 
            ["%.3f" % (intercept / extreme_val[0]) 
                    for intercept in intercepts] if is_normed else
                    ["%.0f" % intercept for intercept in intercepts], 
            loc="upper center", fontsize=fs, 
            title="Boundary NRMS" if is_normed else "Boundary RMS [gal]"
        )
        plt.gca().add_artist(legend_1)
        if sample_num > 1:
            legend_0 = plt.legend([fr_1[0], fr_2[0], fr_3[0]], 
                                  ["Extreme", "Referred", "Ignored"], 
                                  fontsize=fs, loc="upper left", 
                                  title="Sample Points")
            plt.gca().add_artist(legend_2)

    if pic_path:
        plt.savefig(pic_path, bbox_inches="tight", pad_inches=0.1)
    else:
        plt.show()
    plt.close()

def save_result(csv_dir, csv_file, suffix: str, 
                X, 
                train_percent,
                extreme_percent,
                pre_sel_prob, 
                theta_eps,
                thresholds, intercepts, 
                final_theta, MF_pre_sel, 
                A, B, C, N, 
                extreme_val):
    ''' 
    purpose: partitions samples in the training set
    params: csv_dir - csv directory
            csv_file - csv file
            suffix - "train" or "test"
            X - raw data
            train_percent - percent of train data
            extreme_percent - percent of extreme samples
            pre_sel_prob - pre-selection probability
            theta_eps - tolerance for two consecutive theta values
            thresholds - probability thresholds of the MF
            intercepts - value of x when y == 0, in the (RMS, CCD) plain
            final_theta, MF_pre_sel, A, B, C, N - parameters
            extreme_val - [extreme_RMS, extreme_CCD]
    return: None
    '''

    # data 
    RMS = X[:, 0]
    CCD = X[:, 1]
    extreme_RMS = extreme_val[0]
    extreme_CCD = extreme_val[1]
    NRMS = np.minimum(1, RMS / extreme_RMS)
    SRMS = NRMS ** 2 
    NCCD = np.minimum(1, CCD / extreme_CCD)  # Squared Normalized CCD
    SCCD = NCCD ** 2
    X_sqr = np.c_[SRMS, SCCD]

    # get mixed feature
    alpha = np.cos(final_theta)
    beta = np.sin(final_theta)
    MF = get_mixed_feature(X_sqr, (alpha, beta))
    ign_info = (MF < MF_pre_sel)
    Ks_1 = np.array([calc_quantile(MF, threshold) for threshold in thresholds])

    # plot theta_vs_cos diagram for the final_theta
    name = csv_file.rsplit('.', 1)[0]
    # plot_theta_cos(
    #     lambda t: get_dot_algo(*A, *B, *C, N, MF_pre_sel, t),
    #     final_theta, 
    #     os.path.join(csv_dir, f"{name}-theta-{suffix}.jpg")
    # )
    plot_theta_cos_trad_vs_algo(
        X_sqr, 
        pre_sel_prob, 
        lambda t: get_dot_algo(*A, *B, *C, N, MF_pre_sel, t),
        final_theta, 
        os.path.join(csv_dir, f"{name}-theta-{suffix}.jpg")
    )

    # plot classification in the (SRMS, SCCD) plain
    lines = [[-alpha/beta, K/beta, threshold] for (K, threshold) in zip(Ks_1, thresholds)]
    plot_classified_sample(
        X_sqr,  
        os.path.join(csv_dir, f"{name}-sqr-{suffix}.jpg"),
        ign_info, 
        [1., 1.],
        True,
        ["SRMS", "SCCD"],
        lines
    )

    # get curves in the (NRMS, NCCD) plain
    # determined by probability
    curves_3 = [[np.sqrt(K/alpha), np.sqrt(K/beta), 
                 threshold] for (K, threshold) in zip(Ks_1, thresholds)]
    # determined by absolute value
    Ks_2 = alpha * (intercepts / extreme_RMS) ** 2
    curves_4 = [[np.sqrt(K/alpha), np.sqrt(K/beta), float(intercept / extreme_RMS)] \
            for (K, intercept) in zip(Ks_2, intercepts)]

    # # plot classification in the (NRMS, NCCD) plain
    # plot_classified_sample(
    #     np.c_[NRMS, NCCD], 
    #     os.path.join(csv_dir, f"{name}-norm-{suffix}.jpg"),
    #     ign_info, 
    #     [1., 1.], 
    #     is_normed=True,
    #     labels=["NRMS", "NCCD"],
    #     curves=[curves_3, curves_4],  # these curves are in the (NRMS, NCCD) plain
    # )

    # get curves in the (RMS, CCD) plain
    curves_1 = [[extreme_RMS * np.sqrt(K/alpha), extreme_CCD * np.sqrt(K/beta), 
                 threshold] for (K, threshold) in zip(Ks_1, thresholds)]
    curves_2 = [[extreme_RMS * np.sqrt(K/alpha), extreme_CCD * np.sqrt(K/beta), 
            float(intercept)] for (K, intercept) in zip(Ks_2, intercepts)]

    # plot classification in the (RMS, CCD) plain
    plot_classified_sample(
        X, 
        os.path.join(csv_dir, f"{name}-raw-{suffix}.jpg"),
        ign_info, 
        extreme_val, 
        is_normed=False, 
        labels=["RMS [gal]", "CCD"],
        curves=[curves_1, curves_2],  # these curves are in the (RMS, CCD) plain
    )

    # save hyper-parameters and trained results
    data2save = {
        "train_percent": train_percent,
        "extreme_percent": extreme_percent,
        "pre_sel_prob": pre_sel_prob,
        "theta_eps": theta_eps,
        "thresholds": list(thresholds),
        "intercepts": list(intercepts),
        "final_theta": final_theta, 
        "MF_pre_sel": MF_pre_sel,
        "sample_sum": [A, B, C, N], 
        "extreme_val": extreme_val, 
        "lines": lines, 
        "curves_1": curves_1, 
        "curves_2": curves_2
    }
    with open(os.path.join(csv_dir, f"{name}-params-{suffix}.json"), 'w') as f:
        json.dump(data2save, f, indent=4)

def train(csv_dir, 
          ch_name, 
          suffix, 
          train_percent, 
          extreme_percent, 
          pre_sel_prob, 
          theta_eps,
          thresholds,
          intercepts):
    ''' 
    purpose: partitions samples in the training set
    params: csv_dir - csv directory
            ch_name - channel name
            suffix - "train" or "test"
            train_percent - the percent of the train data
            extreme_percent - extreme percent
            pre_sel_prob - pre-selection probability
            theta_eps - tolerance for two consecutive theta values
            thresholds - probability thresholds of the MF
            intercepts - the value of x when y == 0
    return: None
    '''

    # get cable name and ext
    csv_file = f"{ch_name}.csv"

    # get train samples 
    X = get_sample(os.path.join(csv_dir, csv_file))
    sample_num = X.shape[0]
    train_num = int(train_percent * sample_num)
    X_train = X[:train_num]
    RMS = X_train[:, 0]
    CCD = X_train[:, -1]
    
    # Normalized RMS
    num_in_BVec = int(train_num * extreme_percent)  # num of extreme samples
    num_in_ATree = train_num - num_in_BVec
    extreme_RMS = np.sort(RMS)[num_in_ATree]
    NRMS = np.minimum(1, RMS / extreme_RMS)
    SRMS = NRMS ** 2  # Squared Normalized RMS
    # Normalized CCD
    extreme_CCD = np.sort(CCD)[num_in_ATree]
    NCCD = np.minimum(1, CCD / extreme_CCD)  # Squared Normalized CCD
    SCCD = NCCD ** 2

    # # plot the distribution of NRMS, NCCD, SRMS, SCCD
    # plot_distribution(NRMS, os.path.join(csv_dir, name), "NRMS", [0, 1])
    # plot_distribution(NCCD, os.path.join(csv_dir, name), "NCCD", [0, 1])
    # plot_distribution(SRMS, os.path.join(csv_dir, name), "SRMS", [0, 1])
    # plot_distribution(SCCD, os.path.join(csv_dir, name), "SCCD", [0, 1])

    # integrate
    X_sqr = np.c_[SRMS, SCCD]
    
    # # pre-selection
    # iso_info = IsolationForest(max_samples=0.1, contamination=0.001).fit_predict(X_sqr)
    # iso_info = LocalOutlierFactor(contamination=0.01).fit_predict(X_sqr)
    # iso_info = OneClassSVM(nu=0.001, kernel="poly", degree=1, cache_size=2000).fit_predict(X_sqr)
    # iso_info = SGDOneClassSVM(nu=0.001).fit_predict(X_sqr)

    # get main parameters in the sqr plain through iteration
    # final_theta, MF_pre_sel, A, B, C, N = \
    #         iter_theta("", "", X_sqr, pre_sel_prob, theta_eps)
    final_theta = brentq_minimize(
        lambda t: get_dot_trad(X_sqr, t, None, pre_sel_prob), 
        theta_eps
    )
    MF = get_mixed_feature(X_sqr, (np.cos(final_theta), np.sin(final_theta)))
    MF_pre_sel = calc_quantile(MF, pre_sel_prob)
    ign_info = (MF < MF_pre_sel)
    A, B, C, N = get_sample_sum(X_sqr[~ign_info])

    # save results
    save_result(csv_dir, csv_file, suffix, 
                X_train, 
                train_percent,
                extreme_percent,
                pre_sel_prob, 
                theta_eps,
                thresholds, intercepts,  
                final_theta, MF_pre_sel, 
                A, B, C, N, 
                [extreme_RMS, extreme_CCD])

def main():
    # hyper-parameter
    csv_dir = "./data"
    train_percent = 0.75  # [0.75, 0.80, 0.90, 0.95, 0.98, 0.99]
    extreme_percent = 1e-4  # 0.997 -> 0.003=3e-3 -> 3e-4 -> 1e-4
    pre_sel_prob = 0.90
    theta_eps = 1e-4
    thresholds = np.array([0.95, 0.99])
    intercepts = np.array([100., 200.])
    all_channel = ["ch%02d" % (i+1) for i in range(36)]
    # VIV channel: ["ch02", "ch12", "ch20", "ch25", "ch26", "ch27", "ch36"]
    sel_channel = all_channel

    # cpu
    cpu_num = os.cpu_count()
    mw = None
    if cpu_num <= 8:
        mw = 3
    elif cpu_num <= 32:
        mw = 6

    # clear
    for del_file in os.listdir(csv_dir):
        if del_file.rsplit('.', 1)[-1] in ["jpg", "pdf", "json"]:
            os.remove(os.path.join(csv_dir, del_file))

    # train and test-direct
    t0 = time()
    pool = ProcessPoolExecutor(max_workers=mw)  # max_workers=1
    for ch_name in sel_channel:
        pool.submit(train, csv_dir, ch_name, "train", 
                    train_percent, extreme_percent, pre_sel_prob, 
                    theta_eps, thresholds, intercepts)
        pool.submit(train, csv_dir, ch_name, "test-direct", 
                    1., extreme_percent, pre_sel_prob, 
                    theta_eps, thresholds, intercepts)
    pool.shutdown(True)
    print(time() - t0)

if __name__ == "__main__":
    main()
