import os
import json
import numpy as np
from time import time
from scipy.io import loadmat
from concurrent.futures import ProcessPoolExecutor
from sortedcontainers import SortedList, SortedSet
from calc import calc_VIV_s
from train import get_sample, get_dot_algo, fsolve_brentq_minimize, \
        iter_theta, plot_classified_sample, save_result, train

def load_params(csv_dir, ch: str):
    ''' 
    purpose: load trained result
    params: csv_dir - csv directory
            ch - cable name
    return: components of data2save (trained result)
    '''

    with open(os.path.join(csv_dir, f"{ch}-params-train.json"), 'r') as f:
        data2save = json.load(f)

    final_theta   = data2save["final_theta"]
    MF_pre_sel    = data2save["MF_pre_sel"]
    sample_sum    = data2save["sample_sum"]
    extreme_val   = data2save["extreme_val"]
    lines         = data2save["lines"]
    curves_1      = data2save["curves_1"]
    curves_2      = data2save["curves_2"]

    return final_theta, MF_pre_sel, sample_sum, extreme_val, lines, curves_1, curves_2 

def identify(rms, ccd, curves_1, curves_2) -> str:
    ''' 
    purpose: determine the warning level of VIV
    params: rms - single RMS
            ccd - single CCD
            curves_1 - [[a, b, threshold], ...], in (RMS, CCD) plain
            curves_2 - [[a, b, intercept], ...]
    return: warning_level, i.e., "(0, 0)", "(2, 3)", (equivalent to "A1", "C4"...)
    '''

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

    # absolute_level = [chr(ord('A') + i) for i in range(26)]
    # relative_level = [chr(ord('1') + i) for i in range(9)]
    # warning_level = absolute_level[count_abs] + relative_level[count_pro]

    warning_level = "(%d, %d)" % (count_abs, count_pro)
    return warning_level

def judge_VIV(warning_level):
    ''' 
    purpose: judge the occurrence of VIV
    params: warning_level, i.e., "A1", "C4"...
    return: if warning_level == "A1" return 0, else return 1
    '''

    return (warning_level != "A1")

def demo():
    csv_dir = "./data"
    ch = "ch02"
    final_theta, MF_pre_sel, sample_sum, extreme_val, lines, \
            curves_1, curves_2  = load_params(csv_dir, ch)
    print(final_theta, MF_pre_sel, sample_sum, extreme_val, lines, \
            curves_1, curves_2)
    
    fs = 50
    # mat_file = "../../raw-data/Tongling/2022-07-01/2022-07-01 00-VIC.mat"
    mat_file =  "../res/algorithm/2021-04-23 10-VIC.mat"
    a = loadmat(mat_file)["data"][:, int(ch[2:])-1]
    N = a.shape[0]  # 60 min
    acc = a[:N//60]  # 1 min
    error_flag, rms, ccd = calc_VIV_s(acc, fs)
    print(error_flag, rms, ccd)
    if error_flag:
        return "(-1, -1)"
    
    extreme_RMS = extreme_val[0]
    extreme_CCD = extreme_val[1]
    nrms = np.minimum(1, rms / extreme_RMS)
    srms = nrms ** 2
    nccd = np.minimum(1, ccd / extreme_CCD)
    sccd = nccd ** 2

    plot_classified_sample(
        np.array([[srms, sccd]]), 
        None,
        None, 
        [1., 1.], 
        True,
        ["SRMS", "SCCD"],
        lines
    )

    curves_3 = [(a / extreme_RMS, b / extreme_CCD, c) for (a, b, c) in curves_1]
    curves_4 = [(a / extreme_RMS, b / extreme_CCD, c / extreme_RMS) \
                for (a, b, c) in curves_2]
    plot_classified_sample(
        np.array([[nrms, nccd]]), 
        None,
        None, 
        [1., 1.], 
        is_normed=True,
        labels=["NRMS", "NCCD"],
        curves=[curves_3, curves_4],  # these curves are in the (NRMS, NCCD) plain
    )

    plot_classified_sample(
        np.array([[rms, ccd]]),
        None,
        None, 
        extreme_val,
        False, 
        ["RMS [gal]", "NCCD"], 
        curves=[curves_1, curves_2]
    )
    
    warning_level = identify(rms, ccd, curves_1, curves_2)
    print(warning_level)

def get_init_containers(X_train, num_in_ATree):
    ''' 
    purpose: get initial containers
    params: X_train - train samples
            num_in_ATree - num of samples in ATree
    return: ATree_RMS, BVec_RMS, ATree_CCD, BVec_CCD - containers
    '''

    # ATree and BVec for SRMS
    RMS_train = X_train[:, 0]
    sorted_RMS = np.sort(RMS_train)
    # don't use SortedSet, where elements are unrepeatable
    ATree_RMS = SortedList(sorted_RMS[:num_in_ATree])
    BVec_RMS = SortedList(sorted_RMS[num_in_ATree:])
    # ATree and BVec for SCCD
    CCD_train = X_train[:, 1]
    sorted_CCD = np.sort(CCD_train)
    ATree_CCD = SortedList(sorted_CCD[:num_in_ATree])
    BVec_CCD = SortedList(sorted_CCD[num_in_ATree:])
    return ATree_RMS, BVec_RMS, ATree_CCD, BVec_CCD

def modify_container(ATree, BVec, val, extreme_percent, total_samples) -> bool:
    ''' 
    Modify the container and check whether the extreme value changes.

    Parameters
    ----------
    ATree_RMS, BVec_RMS : 
        sorted containers
    old_params : 
        old values
    val : float 
        new rms or ccd
    extreme_percent : float
        extreme percent
    total_samples : int
        all train + part of test, current index in X

    Returns
    -------
    val_flag : bool
        val_flag == True if the extreme value changes, else False
    '''

    # extreme_val = BVec[0]
    num_in_BVec = int(total_samples * extreme_percent)
    ATree_add_one = (num_in_BVec == len(BVec))
    val_flag = False

    if val < ATree[-1]:
        ATree.add(val)
        if not ATree_add_one:
            BVec.add(ATree.pop(-1))
            val_flag = True
    elif val > BVec[0]:
        BVec.add(val)
        if ATree_add_one:
            ATree.add(BVec.pop(0))
            val_flag = True
    else:  # val in [ ATree[-1], BVec[0] ]
        if not ATree_add_one:
            BVec.add(val)
            val_flag = True
        else:
            ATree.add(val)

    return val_flag

def update(X,
           extreme_percent,
           pre_sel_prob,
           theta_eps,
           X_sqr_all,                                 # update
           ATree_RMS, BVec_RMS, ATree_CCD, BVec_CCD,  # update
           old_params,                                # update
           rms, ccd,
           total_samples):
    ''' 
    Updates X_sqr_all, ATree_RMS, BVec_RMS, ATree_CCD, BVec_CCD, old_params.

    Parameters
    ----------
    X : 
        raw data
    extreme_percent : float
        extreme percent
    pre_sel_prob : float
        pre-selection probability
    theta_eps : float
        tolerance for two consecutive theta values
    X_sqr_all : 
        [NRMS**2, NCCD**2], length = num of all the train and test data
    ATree_RMS, BVec_RMS, ATree_CCD, BVec_CCD : 
        sorted containers
    old_params : 
        old values
    rms, ccd : float 
        new sample
    total_samples : int
        all train + part of test, current_index == total_samples - 1
    '''

    old_theta, old_MF_pre_sel, old_sample_sum, old_extreme_val = old_params
    old_A, old_B, old_C, old_N = old_sample_sum

    # modify the container and check whether the extreme value changes
    extreme_RMS_changed = modify_container(ATree_RMS, BVec_RMS, rms, 
                                           extreme_percent, total_samples)
    extreme_CCD_changed = modify_container(ATree_CCD, BVec_CCD, ccd, 
                                           extreme_percent, total_samples)
    
    # change X_sqr_all and old_params
    current_index = total_samples - 1
    if extreme_RMS_changed or extreme_CCD_changed:
        # extreme_val = BVec[0]
        extreme_RMS = BVec_RMS[0]
        extreme_CCD = BVec_CCD[0]
        # change X_sqr_all
        if extreme_RMS_changed:
            X_sqr_all[:current_index, 0] = \
                    np.minimum(1, X[:current_index, 0] / extreme_RMS) ** 2
        if extreme_CCD_changed:
            X_sqr_all[:current_index, 1] = \
                    np.minimum(1, X[:current_index, 1] / extreme_CCD) ** 2
        X_sqr_curr = X_sqr_all[:current_index]
        final_theta, final_MF_pre_sel, final_A, final_B, final_C, final_N = \
                iter_theta("", "", X_sqr_curr, pre_sel_prob, theta_eps, old_theta)
        # change old_params
        old_params[0] = final_theta
        old_params[1] = final_MF_pre_sel
        old_params[2] = [final_A, final_B, final_C, final_N]
        old_params[3] = [extreme_RMS, extreme_CCD]
    else:
        srms = np.minimum(1, rms / old_extreme_val[0]) ** 2
        sccd = np.minimum(1, ccd / old_extreme_val[1]) ** 2
        # change X_sqr_all
        X_sqr_all[current_index] = [srms, sccd]
        # change old_params
        mf = srms * np.cos(old_theta) + sccd * np.sin(old_theta)
        if mf <= old_MF_pre_sel:
            return
        
        # ##############################
        # # demonstrates limited effects
        # s = srms
        # t = sccd
        # a3 = old_A[3] + s*s*s 
        # a2 = old_A[2] + s*s*t 
        # a1 = old_A[1] + s*t*t 
        # a0 = old_A[0] + t*t*t 
        # b2 = old_B[2] + s*s
        # b1 = old_B[1] + s*t
        # b0 = old_B[0] + t*t
        # c1 = old_C[1] + s
        # c0 = old_C[0] + t
        # A = [a3, a2, a1, a0]
        # B = [b2, b1, b0]
        # C = [c1, c0]
        # N = old_N + 1

        # test_dot = lambda t: get_dot_algo(*A, *B, *C, N, old_MF_pre_sel, t)
        # theta_opt = fsolve_brentq_minimize(test_dot, old_theta, theta_eps)

        # if np.abs(theta_opt - old_theta) < theta_eps:
        #     old_params[2] = [A, B, C, N]
        #     return
        # ##############################
        
        X_sqr_curr = X_sqr_all[:current_index]
        # Do not use theta_opt as initial theta, instead, use old_theta!
        final_theta, final_MF_pre_sel, final_A, final_B, final_C, final_N = \
                iter_theta("", "", X_sqr_curr, pre_sel_prob, theta_eps, old_theta)  
        old_params[0] = final_theta
        old_params[1] = final_MF_pre_sel
        old_params[2] = [final_A, final_B, final_C, final_N]

def test(csv_dir, 
         ch_name, 
         suffix, 
         train_percent, 
         extreme_percent, 
         pre_sel_prob, 
         theta_eps,
         intercepts, 
         thresholds):
    ''' 
    purpose: partitions samples in the test set and updates boundary parameters
    params: csv_dir - csv directory
            ch_name - channel name
            suffix - "train" or "test"
            train_percent - the percent of the train data
            extreme_percent - extreme percent
            pre_sel_prob - pre-selection probability
            theta_eps - tolerance for two consecutive theta values
            intercepts - the value of x when y == 0
            thresholds - probability thresholds of the MF
    return: None
    '''

    # get trained parameters
    final_theta, MF_pre_sel, sample_sum, extreme_val, lines, \
            curves_1, curves_2  = load_params(csv_dir, ch_name)
    
    # containers for extreme value and non-extreme value
    csv_file = f"{ch_name}.csv"
    X = get_sample(os.path.join(csv_dir, csv_file))
    sample_num = X.shape[0]
    train_num = int(train_percent * sample_num)
    X_train = X[:train_num]
    num_in_BVec = int(train_num * extreme_percent)
    num_in_ATree = train_num - num_in_BVec
    ATree_RMS, BVec_RMS, ATree_CCD, BVec_CCD = \
            get_init_containers(X_train, num_in_ATree)
    
    # identify and update
    wl = []
    # X_sqr_no_norm2one.shape[0] == num of all the train and test data
    X_sqr_all = np.c_[np.minimum(1, (X[:, 0]/extreme_val[0]))**2, 
                      np.minimum(1, (X[:, 1]/extreme_val[1]))**2]
    old_params = [final_theta, MF_pre_sel, sample_sum, extreme_val]
    for i, (rms, ccd) in enumerate(X[train_num:]):
        if i % 100 == 0:
            print(ch_name, i)
        warning_level = identify(rms, ccd, curves_1, curves_2) 
        wl.append(warning_level)
        # updates X_sqr_all, ATree_RMS, BVec_RMS, ATree_CCD, BVec_CCD, old_params
        total_samples = train_num + (i+1)
        update(
            X,
            extreme_percent,
            pre_sel_prob,
            theta_eps,
            X_sqr_all,
            ATree_RMS, BVec_RMS, ATree_CCD, BVec_CCD, 
            old_params, 
            rms, ccd,
            total_samples
        )
        
    # save results    
    # old_params = [final_theta, MF_pre_sel, sample_sum, extreme_val]
    # old_params[2] == sample_sum == [A, B, C, N]
    save_result(csv_dir, csv_file, suffix, 
                X, 
                old_params[3],
                pre_sel_prob, 
                intercepts, thresholds, 
                old_params[0], old_params[1], *old_params[2])

def main():
    # hyper-parameter
    csv_dir = "./data"
    train_percent = 0.75  # [0.75, 0.8, 0.9, 0.95, 0.98, 0.99]
    extreme_percent = 1e-4
    pre_sel_prob = 0.90
    theta_eps = 1e-4
    intercepts = np.array([100, 200])  
    thresholds = np.array([0.95, 0.97, 0.99])
    all_channel = ["ch%02d" % (i+1) for i in range(36)]
    # VIV channel: ["ch02", "ch12", "ch20", "ch25", "ch26", "ch27", "ch36"]
    sel_channel = all_channel

    # clear
    del_lst = []
    for del_file in os.listdir(csv_dir):
        if del_file.rsplit('.', 1)[-1] in ["jpg", "pdf", "json"]:
            del_lst.append(del_file)
    for del_file in del_lst:
        os.remove(os.path.join(csv_dir, del_file))

    # train and test-direct
    t0 = time()
    pool_train = ProcessPoolExecutor()  # max_workers=1
    for ch_name in all_channel:
        if ch_name not in sel_channel:  # "ch02", "ch04", "ch12", "ch20"
            continue
        pool_train.submit(train, csv_dir, ch_name, "train", 
                          train_percent, extreme_percent, 
                          pre_sel_prob, theta_eps, intercepts, thresholds)
        pool_train.submit(train, csv_dir, ch_name, "test-direct", 
                          1., extreme_percent, 
                          pre_sel_prob, theta_eps, intercepts, thresholds)
    pool_train.shutdown(True)
    print(time() - t0)

    # test-iterate
    t0 = time()
    pool_test = ProcessPoolExecutor()
    for ch_name in all_channel:
        if ch_name not in sel_channel:
            continue
        pool_test.submit(test, csv_dir, ch_name, "test-iterate", 
                         train_percent, extreme_percent, pre_sel_prob, 
                         theta_eps, intercepts, thresholds)
    pool_test.shutdown(True)
    print(time() - t0)

if __name__ == "__main__":
    main()
