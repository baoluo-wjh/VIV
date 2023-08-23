import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from concurrent.futures import ProcessPoolExecutor

plt.rc('font', size=24)
plt.rc('font', family='Times New Roman')
plt.rc('text', usetex = True)

def get_sample(csv_file):
    ''' 
    purpose: get all the right samples (without 9999)
    params: csv_file - csv file that records all the (RMS, HCD)
    return: right_sample - 2d ndarray, right sample
    '''

    sample = pd.read_csv(csv_file).values
    right_index = ((sample[:, 1] < 9.999) & (sample[:, 0] > 0.001))
    right_sample = sample[right_index]
    return right_sample

def get_modified_feature(x, trans="scale", eps_t=6, degree=2):
    ''' 
    purpose: get modified feature
    params: x - original feature
            trans - trans method
            eps_t - eps = 0.1 ** (-eps_t)
            degree - poly degree
    return: x_modi - modified feature
    '''

    x_modi = x.copy()

    if trans == "scale":
        x_modi /= x_modi.max()
    elif trans == "normal":
        sigma = x_modi.std()
        x_modi /= sigma
    elif trans == "log":
        eps = 0.1 ** eps_t
        x_modi = -np.log10(x_modi + eps)
        x_modi[x_modi < 0] = 0
        x_modi /= eps_t
        x_modi = 1 - x_modi
    elif trans == "poly":
        x_modi **= degree

    return x_modi

def get_main_direction(X):
    ''' 
    purpose: get main direction
    params: X - sample
    return: n - main direction
    '''

    # PCA-determined, for VIV cables
    model = PCA(n_components=1).fit(X)
    n = model.components_[0]
    if n[0] < 0 and n[1] < 0:
        n *= -1
    # distribution-determined, for non-VIV cables
    if n[0] * n[1] <= 0:
        n[0] = calc_quantile(X[:, 0], 0.95) - calc_quantile(X[:, 0], 0.05)
        n[1] = calc_quantile(X[:, 1], 0.95) - calc_quantile(X[:, 1], 0.05)
    n /= n.sum()
    return n

def get_mixed_feature(X, main_direction):
    ''' 
    purpose: get mixed feature
    params: X - sample
            main_direction - main direction
    return: mixed_feature - mixed feature
    '''

    alpha, beta = main_direction
    mixed_feature = X[:, 0] * alpha + X[:, 1] * beta
    return mixed_feature

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

def plot_dist(x, prefix, f_name="MF", x_range=None, hist_bins=250, threshold=None):
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

    # plot cdf
    _, ax = plt.subplots(figsize=(20, 10))
    plt.hist(x, bins=hist_bins, color="tab:orange", edgecolor='k', alpha=0.5, 
             density=True, cumulative=True)
    if x_range:
        plt.xlim(x_range)
    if threshold:
        K = calc_quantile(x, threshold)
        plt.plot([0, K], [threshold, threshold], linestyle='--', 
                 linewidth=4, color="tab:blue", alpha=0.5)
        plt.plot([K, K], [threshold, 0], linestyle='--', 
                 linewidth=4, color="tab:blue", alpha=0.5)
        plt.scatter([K], [threshold], s=512, c="tab:blue", marker="*")
        plt.text(K, threshold, f"({K:.3f}, {threshold:.3f})", ha="center", 
                 va="bottom", fontsize=32, fontweight="bold", color="tab:blue")
    plt.xlabel(f_name, size=64)
    plt.ylabel("CDF", size=64)
    plt.xticks(fontsize=48)
    plt.yticks(fontsize=48)
    plt.savefig(f"{prefix}-{f_name}-CDF.jpg", 
                bbox_inches="tight", pad_inches=0.05)
    plt.close()

def plot_classified_sample(X, pic_path, is_scaled=False, 
                           labels=["RMS [gal]", "HCD"], 
                           lines=None, curves=None):
    ''' 
    purpose: plot lined samples 
    params: X - sample
            pic_path - picture path
            is_scaled - whether is modified
            labels - x label and y label
            lines - decision lines
            curves - decision curves
    return: None
    '''

    if is_scaled:
        plt.figure(figsize=(20, 20))
    else:
        plt.figure(figsize=(20, 10))
    plt.plot(X[:, 0], X[:, 1], 'o', markerfacecolor = 'w', 
             markeredgecolor="tab:blue", markersize=4)
    plt.ylim([-0.05, 1.05])
    plt.yticks([0, 0.5, 1], fontsize=48)
    plt.xlabel(labels[0], size=64)
    plt.ylabel(labels[1], size=64)
    if not is_scaled:
        plt.xticks(fontsize=48)
    else:
        plt.axis("equal")
        plt.xlim([-0.05, 1.05])
        plt.xticks([0, 0.5, 1], fontsize=48)
    if lines:
        for (k, b) in lines:
            x = np.linspace(0, 1, 1001, endpoint=True)
            y = k * x + b
            inner_index = ((y >= 0) & (y <= 1))
            if not is_scaled:
                x *= X[:, 0].max()
            plt.plot(x[inner_index], y[inner_index], color="tab:orange", 
                     linestyle="--", linewidth=4)
    if curves:
        for (alpha, beta, eps, L) in curves:
            x = np.linspace(0, 1-eps, 1001, endpoint=True)
            y = -eps + (L*(x+eps)**alpha) ** (-1/beta)
            inner_index = ((y >= 0) & (y <= 1-eps))
            if not is_scaled:
                x *= X[:, 0].max()            
            plt.plot(x[inner_index], y[inner_index], color="tab:green", 
                        linestyle="--", linewidth=4)

    plt.savefig(pic_path, bbox_inches="tight", pad_inches=0.1)
    plt.close()

def classify(csv_dir, csv_file):
    ''' 
    purpose: main process
    params: csv_dir - csv directory
            csv_file - csv file
    return: None
    '''

    # get name and ext
    name = csv_file.rsplit('.', 1)[0]

    # get sample points
    X = get_sample(os.path.join(csv_dir, csv_file))
    RMS = X[:, 0]
    HCD = X[:, 1]

    # plot the distribution of RMS and HCD
    plot_dist(RMS, os.path.join(csv_dir, name), "RMS")
    plot_dist(HCD, os.path.join(csv_dir, name), "HCD")
    
    # get logged sample
    t = 3
    eps = 0.1 ** t
    max_RMS = RMS.max()
    RMS_scale = get_modified_feature(RMS, "scale")
    RMS_log = get_modified_feature(RMS_scale, "log", eps_t=t)
    HCD_log = get_modified_feature(HCD, "log", eps_t=t)

    # integrate
    X_log = np.c_[RMS_log, HCD_log]

    # get main direction in the log plain
    alpha, beta = get_main_direction(X_log)

    # get mixed feature
    mixed_feature = get_mixed_feature(X_log, (alpha, beta))
    thresholds = np.array([0.98])
    Ks = np.array([calc_quantile(mixed_feature, threshold) \
            for threshold in thresholds])

    # plot the distribution of MF
    plot_dist(mixed_feature, os.path.join(csv_dir, name), 
              threshold=thresholds[0])

    # plot classification in the log plain
    plot_classified_sample(
        X_log, 
        os.path.join(csv_dir, f"{name}-classify-log.jpg"),
        True,
        ["TRMS", "THCD"],
        lines=[(-alpha/beta, K/beta) for K in Ks]
    )

    # get lines in the raw plain
    x_intercepts = np.array([100]) / max_RMS
    y_intercepts = np.array([0.3])  # 0.6
    ks = -y_intercepts / x_intercepts
    bs = y_intercepts

    # get curves in the raw plain
    Ls = 10 ** (t * (alpha + beta - Ks))

    # # plot classification in the scale plain
    # plot_classified_sample(
    #     np.c_[RMS_scale, HCD], 
    #     os.path.join(csv_dir, f"{name}-classify-scale.jpg"),
    #     True,
    #     ["SRMS", "HCD"],
    #     lines=zip(ks, bs),
    #     curves=[(alpha, beta, eps, L) for L in Ls],
    # )

    # plot classification in the raw plain
    plot_classified_sample(
        X, 
        os.path.join(csv_dir, f"{name}-classify-raw.jpg"),
        False,
        lines=zip(ks, bs),
        curves=[(alpha, beta, eps, L) for L in Ls],
    )

def main():
    pool = ProcessPoolExecutor()  # max_workers=1
    csv_dir = "./data"
    for csv_file in sorted(os.listdir(csv_dir)):
        name, ext = csv_file.rsplit('.', 1)
        if ext != "csv":
            continue
        # if name not in ["ch04"]:
        #     continue
        pool.submit(classify, csv_dir, csv_file)
    pool.shutdown(True)

if __name__ == "__main__":
    main()
