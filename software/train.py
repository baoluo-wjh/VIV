import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from scipy.optimize import brentq, fsolve
from concurrent.futures import ProcessPoolExecutor
# from sklearn.ensemble import IsolationForest
# from sklearn.neighbors import LocalOutlierFactor
# from sklearn.svm import OneClassSVM
# from sklearn.linear_model import SGDOneClassSVM

np.seterr(all="raise")
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

def input_params_to_test_dot(A, B, C):
    ''' 
    purpose: input parameters to test_dot
    params: A - [a3, a2, a1, a0], a3 = sum(s*s*s), a2 = sum(s*s*t)...
            B - [b2, b1, b0], b2 = sum(s*2), b1 = sum(s*t), b0 = sum(t*t)
            C - [c1, c0], c1 = sum(s), c0 = sum(t)
    return: test_dot - test function
    '''

    a3, a2, a1, a0 = A
    b2, b1, b0 = B
    c1, c0 = C

    def test_dot(theta):
        # vector
        lambd = np.cos(theta)
        mu = np.sin(theta)
        # k = np.array([lambd, mu]).reshape(2, 1)
        # l = np.array([-mu, lambd]).reshape(2, 1)

        # center
        sc = (b2*lambd + b1*mu) / (c1*lambd + c0*mu)
        tc = (b1*lambd + b0*mu) / (c1*lambd + c0*mu)

        # sigma
        sigma_12 = (a2*lambd+a1*mu) - (b2*lambd+b1*mu)*tc - (b1*lambd+b0*mu)*sc + \
                (c1*lambd+c0*mu)*sc*tc
        sigma_11 = (lambd*a3+mu*a2) - 2*(lambd*b2+mu*b1)*sc + (lambd*c1+mu*c0)*sc*sc 
        sigma_22 = (lambd*a1+mu*a0) - 2*(lambd*b1+mu*b0)*tc + (lambd*c1+mu*c0)*tc*tc 
        # Sigma = np.array([[sigma_11, sigma_12], 
        #                   [sigma_12, sigma_22]])

        # unweighted covariance
        # xc = (XTX @ k) / (eTX @ k)
        # Sigma = XTX - xc @ eTX - (xc @ eTX).T + n * xc @ xc.T

        # orthogonal
        # a = l
        # b = Sigma @ k
        # cos_phi = (a.T @ b) / (np.linalg.norm(a) * np.linalg.norm(b))
        # return cos_phi[0, 0]
        b = np.array([sigma_11*lambd+sigma_12*mu, sigma_12*lambd+sigma_22*mu])
        b /= np.sqrt((b**2).sum())
        cos_phi = -mu*b[0] + lambd*b[1]
        return cos_phi
    return test_dot

def get_sample_sum(X):
    ''' 
    purpose: get sample sum
    params: X - sample
    return: A - [a3, a2, a1, a0], a3 = sum(s*s*s), a2 = sum(s*s*t)...
            B - [b2, b1, b0], b2 = sum(s*2), b1 = sum(s*t), b0 = sum(t*t)
            C - [c1, c0], c1 = sum(s), c0 = sum(t)
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
    return A, B, C

def plot_theta_cos(test_dot, theta_opt, pic_path):
    ''' 
    purpose: plot all theta within [0, pi/2]
    params: test_dot - function pointer: input theta, output cos
            theta_opt - optimized theta
            pic_path - picture path
    return: None
    '''

    theta = np.linspace(0, np.pi/2, 1000, endpoint=True)
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
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))
    plt.scatter([theta_opt], [0], s=512, c="tab:orange", marker="*")
    if theta_opt >= np.pi / 4:
        plt.text(theta_opt - 0.03, 0, 
                 "$%.4f \\; (%.4f \\pi)$" % (theta_opt, theta_opt / np.pi), 
                 ha="right", va="center", fontsize=64, color="tab:green")
    else:
        plt.text(theta_opt + 0.03, 0, 
                 "$%.4f \\; (%.4f \\pi)$" % (theta_opt, theta_opt / np.pi), 
                 ha="left", va="center", fontsize=64, color="tab:green")
    plt.savefig(pic_path, bbox_inches="tight", pad_inches=0.05)
    plt.close()

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

def iter_theta(csv_dir, name, X_sqr, pre_sel_prob, init_pre_theta=np.pi/4):
    ''' 
    purpose: iteration to find the final theta (pre_theta = theta_opt)
    params: csv_dir - csv directory
            name - cable name
            X_sqr - squared data
            pre_sel_prob - probability threshold for pre-selection
            init_pre_theta - initial theta for sample pre-selection 
    return: final_theta - final theta
            final_MF_pre_sel - MF value for sample pre-selection
            final_A, final_B, final_C - sample sums
    '''

    theta_opt_lst = [init_pre_theta]
    MF_pre_sel_lst = []
    A_lst = []
    B_lst = []
    C_lst = []
    count_iter = 1
    iter_eps = 1e-4

    while True:
        # pre-selection
        pre_theta = theta_opt_lst[-1]
        MF = get_mixed_feature(X_sqr, (np.cos(pre_theta), np.sin(pre_theta)))
        MF_pre_sel = calc_quantile(MF, pre_sel_prob)
        MF_pre_sel_lst.append(MF_pre_sel)
        iso_info = (MF > MF_pre_sel)
        A, B, C = get_sample_sum(X_sqr[iso_info])
        A_lst.append(A)
        B_lst.append(B)
        C_lst.append(C)

        # find the optimal theta
        test_dot = input_params_to_test_dot(A, B, C)
        # theta_opt = brentq(test_dot, 0, np.pi/2, xtol=iter_eps)
        theta_opt = fsolve(test_dot, x0=pre_theta, xtol=iter_eps)[0]

        # plot theta_vs_cos diagram
        if csv_dir and name:
            pic_path = os.path.join(csv_dir, "%s-%d-pre_theta=%.4fpi.jpg" \
                                    % (name, count_iter, pre_theta/np.pi))
            plot_theta_cos(test_dot, theta_opt, pic_path)
        
        # append current theta_opt to theta_opt_lst
        if count_iter <= 5:
            theta_opt_lst.append(theta_opt)
        else:  
            # double fixed points, very rare, every important!
            theta_opt_lst.append((theta_opt + pre_theta) / 2)
        if np.abs(theta_opt - pre_theta) < iter_eps:
            break
        else:
            count_iter += 1

    final_theta = theta_opt_lst[-1]
    final_MF_pre_sel = MF_pre_sel_lst[-1]
    final_A = A_lst[-1]
    final_B = B_lst[-1]
    final_C = C_lst[-1]
    return final_theta, final_MF_pre_sel, final_A, final_B, final_C

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
            x = np.linspace(0, 1, 1001, endpoint=True)
            y = k * x + b
            inner_index = ((y >= 0) & (y <= 1))
            color_used = '#00'+str(hex(int(0xff*(count/num_line))))[2:]+'ff'
            fr = plt.plot(x[inner_index], y[inner_index], color=color_used, 
                          linestyle="--", linewidth=4)
            leg_1.append(fr[0])
            thresholds.append(threshold)
            if count == 1:
                a = -b / k
                length = np.sqrt(a ** 2 + b ** 2)
                x0 = a / 2
                y0 = b / 2
                cos_ = b / length
                sin_ = a / length
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
                                  ["extreme", "referred", "ignored"], 
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
                x = np.linspace(0, a, 1001, endpoint=True)
                y = b * np.sqrt(1 - (x/a) ** 2)
                if is_normed:
                    inner_index = (x <= 1) & (y <= 1)
                else:
                    inner_index = y <= np.maximum(extreme_val[1], X[:, 1].max())
                xx = x[inner_index]
                yy = y[inner_index]
                color_used ='#00'+str(hex(int(0xff*(count/num_curve))))[2:]+'ff'
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
                    plt.plot([mark], [0.], '*', markersize=32, 
                             markerfacecolor='w', markeredgecolor=color_used)
                    if is_normed:
                        plt.text(mark, -0.02, "%.3f" % mark, ha="center", 
                                 va="top", fontsize=32, color=color_used)
                    else:
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
                                  ["extreme", "referred", "ignored"], 
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
                extreme_val,
                intercepts, thresholds, 
                final_theta, MF_pre_sel, A, B, C):
    ''' 
    purpose: partitions samples in the training set
    params: csv_dir - csv directory
            csv_file - csv file
            suffix - "train" or "test"
            X - raw data
            extreme_val - [extreme_RMS, extreme_CCD]
            intercepts - value of x when y == 0, in the (RMS, CCD) plain
            thresholds - probability thresholds of the MF
            final_theta, MF_pre_sel, A, B, C - parameters
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
    ign_info = (MF <= MF_pre_sel)
    Ks_1 = np.array([calc_quantile(MF, threshold) for threshold in thresholds])

    # plot theta_vs_cos diagram for the final_theta
    name = csv_file.rsplit('.', 1)[0]
    plot_theta_cos(input_params_to_test_dot(A, B, C), final_theta, 
            os.path.join(csv_dir, f"{name}-theta-{suffix}.jpg"))

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

    # plot classification in the (NRMS, NCCD) plain
    plot_classified_sample(
        np.c_[NRMS, NCCD], 
        os.path.join(csv_dir, f"{name}-norm-{suffix}.jpg"),
        ign_info, 
        [1., 1.], 
        is_normed=True,
        labels=["NRMS", "NCCD"],
        curves=[curves_3, curves_4],  # these curves are in the (NRMS, NCCD) plain
    )

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

    # save trained results
    sample_sum = [A, B, C]
    data2save = {"final_theta": final_theta, 
                 "MF_pre_sel": MF_pre_sel,
                 "sample_sum": sample_sum, 
                 "extreme_val": extreme_val, 
                 "lines": lines, 
                 "curves_1": curves_1, 
                 "curves_2": curves_2,}
    with open(os.path.join(csv_dir, f"{name}-params-{suffix}.json"), 'w') as f:
        json.dump(data2save, f, indent=4)

def train(csv_dir, 
          ch_name, 
          suffix, 
          train_percent=0.8, 
          extreme_percent=1e-4, 
          pre_sel_prob=0.95, 
          intercepts=np.array([100, 200]), 
          thresholds=np.array([0.95, 0.99])):
    ''' 
    purpose: partitions samples in the training set
    params: csv_dir - csv directory
            ch_name - channel name
            suffix - "train" or "test"
            train_percent - the percent of the train data
            extreme_percent - extreme percent
            pre_sel_prob - pre-selection probability
            intercepts - the value of x when y == 0
            thresholds - probability thresholds of the MF
    return: None
    '''

    # get cable name and ext
    name = ch_name
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

    # get main direction in the sqr plain through iteration
    final_theta, MF_pre_sel, A, B, C = iter_theta("", "", X_sqr, pre_sel_prob)

    # save results
    save_result(csv_dir, csv_file, suffix, 
                X, 
                [extreme_RMS, extreme_CCD],
                intercepts, thresholds, 
                final_theta, MF_pre_sel, A, B, C)

def main():
    csv_dir = "./data"
    suffix = "train"
    train_percent = 0.75
    extreme_percent = 1e-4
    pre_sel_prob = 0.95
    thresholds = np.array([0.97, 0.99])
    intercepts = np.array([100, 200])  
    pool = ProcessPoolExecutor()  # max_workers=1
    for ch_name in ["ch%02d" % (i+1) for i in range(36)]:
        if ch_name not in ["ch02"]:  # "ch04", "ch12", "ch20"
            continue
        pool.submit(train, csv_dir, ch_name, suffix, train_percent, \
                    extreme_percent, pre_sel_prob, intercepts, thresholds)
    pool.shutdown(True)

if __name__ == "__main__":
    main()
    