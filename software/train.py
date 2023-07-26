import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import norm, gamma
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV

plt.rc('font', size=24)
plt.rc('font', family='Times New Roman')
plt.rc('text', usetex = True)

# root mean square: RMS
# hollow coefficient of Derivative analytical signal: HCD (SRD)

def get_right_sample(csv_file):
    ''' 
    purpose: get all the right samples (without 9999)
    params: csv_file - csv file that records all the (RMS, HCD)
    return: right_sample - 2d ndarray, right sample
    '''

    # RMS_2d shape: (num_time_win * num_mat, num_channel)
    sample = pd.read_csv(csv_file).values
    right_index = ((sample[:, 1] < 9.999) & (sample[:, 0] > 0.001))
    right_sample = sample[right_index]
    return right_sample

def get_modified_sample(X, trans="identity", degree=2):
    ''' 
    purpose: get modified sample
    params: X - original sample
    return: X_modi - modified sample
    '''

    X_modi = X.copy()
    X_modi[:, 0] /= X_modi[:, 0].max()

    if trans == "identity":
        pass
    elif trans == "log":
        eps = 0.001
        X_modi = -np.log10(X_modi + eps)
        X_modi[:, 0][X_modi[:, 0] < 0] = 0
        X_modi /= X_modi.max(axis=0)
    elif trans == "poly":
        X_modi **= degree

    return X_modi

def plot_sample(X, pic_path, is_modified=False, 
                labels=["RMS [mg]", "HCD"]):
    ''' 
    purpose: plot all the sample points
    params: X - sample
            pic_path - picture path
            is_modified - whether is modified
            labels - x label and y label
    return: None
    '''

    if is_modified:
        plt.figure(figsize=(20, 20))
    else:
        plt.figure(figsize=(20, 10))

    plt.plot(X[:, 0], X[:, 1], 'o', markerfacecolor = 'w', 
             markeredgecolor='g', markersize=4)
    plt.ylim([-0.05, 1.05])
    plt.yticks([0, 1], fontsize=48)
    plt.xlabel(labels[0], size=64)
    plt.ylabel(labels[1], size=64)
    if not is_modified:
        plt.xticks(fontsize=48)
    else:
        plt.xlim([-0.05, 1.05])
        plt.xticks([0, 1], fontsize=48)

    plt.savefig(pic_path, bbox_inches="tight", pad_inches=0.1)
    plt.close()

def bi_dbscan_cluster(X, eps_dist=0.01, ratio_1=0.002, ratio_2=0.005):
    ''' 
    purpose: cluster all the samples using DBSCAN
    params: X - modified sample
            eps_dist - DBSCAN cluster distance
            ratio_1 - sampling ratio in the level 1
            ratio_2 - sampling ratio in the level 2
    return: y - clustering result: -1 - VIV, 0 - normal, 1 - high RMS and low HCD
    '''

    # first level cluster
    sample_num = X.shape[0]
    model_1 = DBSCAN(eps=eps_dist, min_samples=int(ratio_1*sample_num))
    result_1 = model_1.fit_predict(X)
    # find points with low RMS and low HCD
    normal_index = -2  # initialization
    normal_R2_avg = 2
    for index in range(-1, int(result_1.max())+1):
        this_type = X[result_1 == index]
        if this_type.shape[0] == 0:
            continue
        # R_avg = np.sqrt(this_type[:, 0] ** 2 + this_type[:, 1] ** 2).mean()
        R2_avg = (this_type ** 2).mean()
        if R2_avg < normal_R2_avg:
            normal_R2_avg = R2_avg
            normal_index = index

    # second level cluster
    rest_X = X[result_1 != normal_index]
    rest_num = rest_X.shape[0]
    model_2 = DBSCAN(eps=eps_dist, min_samples=int(ratio_2*rest_num))
    result_2 = model_2.fit_predict(rest_X)
    # find points with high RMS and low HCD
    hRlH_index = -2  # hRlH: high RMS low HCD
    hRlH_avg = -1
    for index in range(-1, int(result_2.max()+1)):
        this_type = rest_X[result_2 == index]
        if this_type.shape[0] == 0:
            continue
        avg = (this_type[:, 0] / (this_type[:, 1] + 0.001)).mean()
        if avg > hRlH_avg:
            hRlH_avg = avg
            hRlH_index = index

    # y: -1 - VIV, 0 - normal, 1 - high RMS and low HCD
    y = np.zeros_like(result_1, dtype=np.int32)
    rest_y = np.ones(rest_X.shape[0])
    rest_y[result_2 != hRlH_index] = -1
    y[result_1 != normal_index] = rest_y

    # return
    return y

def hdbscan_cluster(X, min_pts_ratio=0.00005):
    ''' 
    purpose: cluster all the samples using HDBSCAN
    params: X - modified sample
            min_pts_ratio - sampling ratio of HDBSCAN
    return: y - clustering result: -1 - VIV, 0 1 2... - normal
    '''

    sample_num = X.shape[0]

    # HDBSCAN
    model = HDBSCAN(min_cluster_size=int(min_pts_ratio*sample_num)).fit(X)
    y = model.labels_

    # return
    return y

def iforest_cluster(X, noise_ratio=0.1):
    ''' 
    purpose: cluster all the samples using IsolationForest
    params: X - modified sample
    return: y - clustering result: -1 - VIV, 1 - normal
    '''

    # IsolationForest
    model = IsolationForest(contamination=noise_ratio, 
                            n_estimators=1000,
                            max_samples=0.001, 
                            n_jobs=-1).fit(X)
    y = model.predict(X)

    # return
    return y

def get_bi_label(y):
    ''' 
    purpose: reset result y
    params: y - clustering result: -1 - VIV, other - normal
    return: new_y - modified clustering result: -1 - VIV, 1 - normal
    '''

    new_y = np.ones_like(y)
    new_y[y == -1] = -1
    return new_y

def get_boundary(X, y, regulation=1):
    ''' 
    purpose: get boundary by LinearSVC
    params: X - normalized sample
            y - clustering result: -1 - VIV, 1 - normal
            regulation - C
    return: (a, b, c) - line: ax + by + c = 0
    '''

    model = LinearSVC(C=regulation, class_weight="balanced", 
                      max_iter=10000, dual=False).fit(X, y)
    
    a = model.coef_[0][0]
    b = model.coef_[0][1]
    c = model.intercept_[0]

    return (a, b, c)
    
def plot_classified_sample(X, y, pic_path, is_two_clusters=True, is_modified=False, 
                           labels=["RMS [mg]", "HCD"], boundry=None):
    ''' 
    purpose: plot classified samples (maybe together with the boundry),
             attention, classification is based on clustering, while boundry 
             is calculated by LinearSVC
    params: X - sample
            y - clustering result: -1 - VIV, all the other clusters - normal
            pic_path - picture path
            is_two_clusters - whether is totally two clusters
            is_modified - whether is modified
            labels - x label and y label
            boundry - (a, b, c)
    return: None
    '''

    if is_modified:
        _, ax = plt.subplots(figsize=(20, 20))
    else:
        _, ax = plt.subplots(figsize=(20, 10))

    unique_y = np.unique(y)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, unique_y.shape[0])]
    for k, col in zip(unique_y, colors):
        class_index = (y == k)
        if not is_two_clusters:
            ax.plot(X[class_index, 0], X[class_index, 1], 'x' if k == -1 else 'o', 
                    markerfacecolor = tuple([0, 0, 0, 1] if k == -1 else col), 
                    markeredgecolor='k', markersize=4)
        else:
            ax.plot(X[class_index, 0], X[class_index, 1], 'o', 
                    markerfacecolor='w',
                    markeredgecolor = 'g' if k == -1 else 'r', markersize=4)
    if not is_two_clusters:
        n_clusters_ = unique_y.shape[0] - (1 if -1 in unique_y else 0)
        ax.set_title(f"Number of clusters: {n_clusters_}", fontsize=64)
    ax.set_ylim([-0.05, 1.05])
    plt.yticks([0, 1], fontsize=48)
    ax.set_xlabel(labels[0], size=64)
    ax.set_ylabel(labels[1], size=64)
    if not is_modified:
        plt.xticks(fontsize=48)
    else:
        ax.set_xlim([-0.05, 1.05])
        plt.xticks([0, 1], fontsize=48)
    if boundry:
        a, b, c = boundry
        if not is_modified:
            a /= X[:, 0].max()
        x = np.linspace(0, X[:, 0].max(), 1001, endpoint=True)
        ax.plot(x, -a/b*x-c/b, color="tab:orange", linestyle="--", linewidth=2)
    plt.savefig(pic_path, bbox_inches="tight", pad_inches=0.1)
    plt.close()

def cluster_method():
    csv_dir = "./src"
    for file in sorted(os.listdir(csv_dir)):
        name, ext = file.rsplit('.', 1)
        if ext == "csv" and name in ["ch02"]:  # "ch12", "ch20"
            #############  get samples
            # get sample points
            X = get_right_sample(os.path.join(csv_dir, file))
            plot_sample(X, os.path.join(csv_dir, f"{name}-raw.jpg"))

            # get normalized sample
            X_scaled = get_modified_sample(X)

            # get squared sample
            X_squared = get_modified_sample(X, "poly", degree=2)
            plot_sample(X_squared, os.path.join(csv_dir, f"{name}-sqr.jpg"), True,
                        ["Squared Normalized RMS", "Squared HCD"])

            #############  identification with clustering
            # # two-level DBSCAN
            # y_dbscan = bi_dbscan_cluster(X_scaled)
            # plot_classified_sample(X, y_dbscan, os.path.join(
            #         csv_dir, f"{name}-raw-DBSCAN.jpg"))
            # y_squared_dbscan = bi_dbscan_cluster(X_squared)
            # plot_classified_sample(X_squared, y_squared_dbscan, os.path.join(
            #         csv_dir, f"{name}-sqr-DBSCAN.jpg"), False, True,
            #         ["Squared Normalized RMS", "Squared HCD"])

            # # HDBSCAN
            # y_hdbscan = hdbscan_cluster(X_scaled)
            # plot_classified_sample(X, y_hdbscan, os.path.join(
            #         csv_dir, f"{name}-raw-HDBSCAN.jpg"))
            # y_squared_hdbscan = hdbscan_cluster(X_squared)
            # plot_classified_sample(X_squared, y_squared_hdbscan, os.path.join(
            #         csv_dir, f"{name}-sqr-HDBSCAN.jpg"), False, True,
            #         ["Squared Normalized RMS", "Squared HCD"])

            # IsolationForest
            y_iforest = iforest_cluster(X_scaled)
            plot_classified_sample(X, y_iforest, os.path.join(
                    csv_dir, f"{name}-raw-IForest.jpg"))
            y_squared_iforest = iforest_cluster(X_squared)         
            plot_classified_sample(X_squared, y_squared_iforest, os.path.join(
                    csv_dir, f"{name}-sqr-IForest.jpg"), True, True,
                    ["Squared Normalized RMS", "Squared HCD"])
            
            # SVDD
            # to be continued...

            #############  get boundary
            # modified y
            y_iforest = get_bi_label(y_iforest)
            # svc
            boundry = get_boundary(X_scaled, y_iforest)
            plot_classified_sample(X, y_iforest, os.path.join(
                    csv_dir, f"{name}-raw-IForest-SVR.jpg"), True, False, 
                    ["RMS [mg]", "HCD"], boundry)
            pd.DataFrame(boundry).to_csv(os.path.join(csv_dir, f"{name}-abc.csv"), 
                                         header=None, index=None)

def get_main_direction(X):
    ''' 
    purpose: get main direction
    params: X - sample
    return: n_0 - main direction
    '''

    model = PCA(n_components=1).fit(X)
    n_0 = model.components_[0]
    return n_0

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

def analyze_GaussianMixture(x, prefix):
    ''' 
    purpose: get the most suitable Gaussian mixture model, 
             plot (aic, bic), 
             plot pdf and cdf
    params: x - mixed feature
            prefix - path and channel
    return: (weights, means, covariances) - parameters of Gaussian mixture model
    '''

    # reshape
    X = x.reshape(-1, 1)

    # get aic and bic (both negative)
    n_list = list(range(1, 10))  # [1, 9]
    aic_list = []
    for i in n_list:
        gmm = GaussianMixture(n_components=i).fit(X)
        aic_list.append(gmm.aic(X))
    neg_aic_array = -np.array(aic_list)
    bic_list = []
    for i in n_list:
        gmm = GaussianMixture(n_components=i).fit(X)
        bic_list.append(gmm.bic(X))
    neg_bic_array = -np.array(bic_list)
    neg_avg_array = (neg_aic_array + neg_bic_array) / 2

    # plot -aic and -bic
    num = len(n_list)
    ind = np.arange(0, num * 3, 3)  # the x locations for the groups
    width = 1.0  # the width of the bars
    fig, ax = plt.subplots(figsize=(20, 8))
    labels = ["$-$AIC", "$-$BIC"]
    bar_1 = ax.bar(ind + (0 - 1) * width, neg_aic_array, width, 
                   color="tab:blue", align='edge')
    bar_2 = ax.bar(ind + (1 - 1) * width, neg_bic_array, width, 
                   color="tab:orange", align='edge')
    ax.scatter(ind, neg_avg_array, marker='*', s=8, c="tab:green")
    ax.plot(ind, neg_avg_array, linestyle='--', linewidth=4, color="tab:green")
    plt.xticks(ind, n_list, fontsize=32)
    plt.yticks(fontsize=32)
    plt.xlabel("Number of Components", size=48)
    plt.ylabel("Criterion", size=48)
    plt.ticklabel_format(axis='y', scilimits=[0, 3])
    plt.legend([bar_1, bar_2], labels, loc='upper left', fontsize=32, 
            frameon=True, framealpha=0.5)
    plt.savefig(f"{prefix}-criterion", bbox_inches='tight', pad_inches=0.1)
    plt.close()

    # select the best n_components
    best_n = n_list[np.argmax(neg_avg_array)]
    if best_n == 9:
        best_n = 0
        for i in range(1, len(n_list)):
            if (neg_avg_array[i] - neg_avg_array[i - 1]) / neg_avg_array[i - 1] <= 0.05:
                best_n = n_list[i - 1]
                break
        if best_n == 0:
            best_n = 3

    # get best model
    best_gmm = GaussianMixture(n_components=best_n).fit(X)
    weights = best_gmm.weights_
    means = best_gmm.means_.flatten()
    covariances = best_gmm.covariances_.flatten()

    # sort weights, from large to small
    seq_index = np.argsort(-weights)
    weights = weights[seq_index]
    means = means[seq_index]
    covariances = covariances[seq_index]

    # return 
    return (weights, means, covariances)


class MixedGaussianDistribution(object):
    """
    encapsulate pdf and cdf
    """

    def __init__(self, weights, means, covariances):
        self.weights = weights
        self.means = means
        self.covariances = covariances
        self.N_list = [norm(means[i], covariances[i]) for i in range(weights.shape[0])]

    def pdf(self, x):
        ''' 
        purpose: get probability density function (PDF)
        params: x - an array of random variable
        return: y - probability density
        '''

        y = 0
        # for (w, m, c) in zip(self.weights, self.means, self.covariances):
        #     y += w * 1/np.sqrt(2*np.pi*c**2) * np.exp(-1/2*((x-m)/c)**2)
        for (w, N) in zip(self.weights, self.N_list):
            y += w * N.pdf(x)
        return y
    
    def cdf(self, x):
        ''' 
        purpose: get cumulative density function (CDF)
        params: x - an array of random variable
        return: y - cumulative density
        '''

        y = 0
        for (w, N) in zip(self.weights, self.N_list):
            y += w * N.cdf(x)
        return y


def get_threshold_MF(x, prefix, gmm_params, proba=0.95):
    ''' 
    purpose: get threshold MF
    params: x - mixed feature
            prefix - path and channel
            gmm_params - (weights, means, covariances)
            proba - threshold probability
    return: threshold_MF - threshold MF
    '''

    weights, means, covariances = gmm_params
    mgd = MixedGaussianDistribution(weights, means, covariances)
    x_mgd = np.linspace(0, x.max(), 1001, endpoint=True)

    # plot pdf
    _, ax = plt.subplots(figsize=(20, 10))
    plt.hist(x, bins=250, color="tab:blue", edgecolor='w', density=True)
    # plt.plot(x_mgd, mgd.pdf(x_mgd), linestyle='-', linewidth=4, color="tab:red")
    text_str = "$n = {%d}; MF \\sim \\sum_{i=1}^{n} \\alpha_{i} \\mathcal{N} \
            \\left( \\mu_{i}, \\sigma_{i} \\right)$ \n" % (weights.shape[0])
    params_line = ["$\\alpha_{%d} = %.4f; \\mu_{%d} = %.4f; \\sigma_{%d} = %.4f$" 
                   % (i+1, weights[i], i+1, means[i], i+1, covariances[i]) 
                   for i in range(weights.shape[0])]
    text_str += '\n'.join(params_line)
    plt.text(0.4, 0.5, text_str, color='tab:green', fontsize=32, 
             fontweight='bold', transform=ax.transAxes)
    plt.xlabel("MF", size=64)
    plt.ylabel("PDF", size=64)
    plt.xticks(fontsize=48)
    plt.yticks(fontsize=48)
    plt.savefig(f"{prefix}-PDF.jpg", bbox_inches="tight", pad_inches=0.1)
    plt.close()

    # plot cdf
    _, ax = plt.subplots(figsize=(20, 10))
    plt.hist(x, bins=50, color="tab:blue", edgecolor='w', 
             density=True, cumulative=True)
    plt.plot(x_mgd, mgd.cdf(x_mgd), linestyle='-', linewidth=4, color="tab:red")
    plt.plot(x_mgd, np.ones_like(x_mgd), linestyle='--', linewidth=4, 
             color="tab:red", alpha=0.5)
    plt.xlabel("MF", size=64)
    plt.ylabel("CDF", size=64)
    plt.xticks(fontsize=48)
    plt.yticks(fontsize=48)
    plt.savefig(f"{prefix}-CDF.jpg", bbox_inches="tight", pad_inches=0.1)
    plt.close()

    # get threshold MF according to proba


def get_intercepts():
    ''' 
    purpose: get intercepts using Gaussian mixture models
    params: x - mixed feature
    return: intercepts - a list of intercept
    '''

def plot_lined_sample(X, pic_path, is_modified=False, 
                      labels=["RMS [mg]", "HCD"], lines=None):
    ''' 
    purpose: plot lined samples 
    params: X - sample
            pic_path - picture path
            is_modified - whether is modified
            labels - x label and y label
            lines - decision lines
    return: None
    '''

    if is_modified:
        plt.figure(figsize=(20, 20))
    else:
        plt.figure(figsize=(20, 10))
    plt.plot(X[:, 0], X[:, 1], 'o', markerfacecolor = 'w', 
             markeredgecolor='g', markersize=4)
    plt.ylim([-0.05, 1.05])
    plt.yticks([0, 1], fontsize=48)
    plt.xlabel(labels[0], size=64)
    plt.ylabel(labels[1], size=64)
    if not is_modified:
        plt.xticks(fontsize=48)
    else:
        plt.xlim([-0.05, 1.05])
        plt.xticks([0, 1], fontsize=48)
    if lines:
        for (k, b) in lines:
            x = np.linspace(0, X[:, 0].max(), 1001, endpoint=True)
            y = k * x + b
            inner_index = ((y >= 0) & (y <= 1))
            plt.plot(x[inner_index], y[inner_index], color="tab:blue", 
                     linestyle="--", linewidth=4)
    plt.savefig(pic_path, bbox_inches="tight", pad_inches=0.1)
    plt.close()

def line_method():
    csv_dir = "./src"
    for file in sorted(os.listdir(csv_dir)):
        name, ext = file.rsplit('.', 1)
        if ext == "csv" and name in ["ch02"]:  # "ch12", "ch20"
            # get sample points
            X = get_right_sample(os.path.join(csv_dir, file))

            # get scaled sample
            X_scaled = get_modified_sample(X)

            # get main direction
            alpha, beta = get_main_direction(X_scaled)

            # get slop
            k = -alpha / beta

            # get mixed feature
            mixed_feature = get_mixed_feature(X_scaled, (alpha, beta))

            # get best GaussianMixture
            # weights, means, covariances = analyze_GaussianMixture(
            #         X_scaled[:, 1], os.path.join(csv_dir, name))

            # # get threshold MF
            threshold_MF = get_threshold_MF(mixed_feature, 
                    os.path.join(csv_dir, name),
                    (weights, means, covariances), proba=0.95)

            # # get intercepts
            # intercepts = get_intercepts(mixed_feature)  # [0.3, 0.6]
            intercepts = [0.3, 0.6]

            # # plot
            plot_lined_sample(X_scaled, os.path.join(csv_dir, f"{name}-norm.jpg"), 
                              True, ["Normalized RMS", "HCD"], 
                              lines=[(k, b) for b in intercepts])
            # plot_lined_sample(X, os.path.join(csv_dir, f"{name}-raw.jpg"), 
            #                   False, ["RMS [mg]", "HCD"], 
            #                   lines=[(k / X[:, 0].max(), b) for b in intercepts])
            
def main():
    # cluster_method()
    line_method()

if __name__ == "__main__":
    main()
