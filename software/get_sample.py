import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from time import time
from concurrent.futures import ProcessPoolExecutor
from scipy.io import loadmat
from h5py import File
from calc import calc_VIV

plt.rc('font', family='Times New Roman')
plt.rc('text', usetex = True)

features = ["RMS", "PRS", "HCH", "HCD", "CCH", "CCD", "MPRS", "MHCH", "AMHCH"]
feature_num = len(features)

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

def cope_all_mat(mat_list, fs, app_fund_arr, minute_interval=5):
    ''' 
    purpose: cope with all mat files by multiprocessing, 
             export RMS and HCD as csv files
    params: mat_list - all mat files
            fs - sampling frequency
            app_fund_arr - array of approximate fundamental frequency
            minute_interval - time interval in minutes
    return: None
    '''

    pool = ProcessPoolExecutor()  # max_workers=1
    for file_path in mat_list:  
        # "../HangZhou/south/2022-12-31 18-VIC.mat"
        # date = file_path.rsplit('/', 1)[-1].split(' ')[0]
        # if date != "2022-07-02":
        #     continue
        # if date.split('-')[1] != "01":
        #     continue
        pool.submit(cope_mat, file_path, fs, app_fund_arr, minute_interval)
    pool.shutdown(True)

def cope_mat(file_path, fs, app_fund_arr=None, minute_interval=5):
    ''' 
    purpose: cope with a mat file, export RMS and HCD as csv files
    params: file_path - mat file's path
            fs - sampling frequency
            app_fund_arr - array of approximate fundamental frequency
            minute_interval - time interval in minutes
    return: None
    '''

    mat_dir, mat_name = os.path.split(file_path)
    if os.path.exists(os.path.join(mat_dir, 
            f"{mat_name.rsplit('-', 1)[0]}_RMS.csv")): return
    
    try:
        acc_2d = loadmat(file_path)["data"].astype(np.float64)  # (50*60*60=180000, 36)
    except:
        acc_2d = np.transpose(File(file_path, 'r')["data"]).astype(np.float64)

    num_cable = acc_2d.shape[1]

    if app_fund_arr is None:
        app_fund_arr = np.full((num_cable,), 0.5)  # default app_fund: 0.5 Hz
    
    # feature_2d_lst[0] is RMS_2d: [[] for _ in range(num_cable)]
    feature_2d_lst = [[[] for _ in range(num_cable)] for _ in range(feature_num)]

    for cable_index in range(num_cable):  # every cable in 60 min
        acc = acc_2d[:, cable_index]
        try:
            cope_cable(cable_index, app_fund_arr[cable_index], 
                       fs, minute_interval, acc, feature_2d_lst)
        except:
            print(f"mat_name: {mat_name}")

    # save RMS, PRS...
    for (f_2d, ft) in zip(feature_2d_lst, features):
        pd.DataFrame(f_2d).to_csv(os.path.join(mat_dir, 
                f"{mat_name.rsplit('-', 1)[0]}_{ft}.csv"), header=None, index=None)

    print(f"{mat_name} finished")

def cope_cable(cable_index, app_fund, fs, minute_interval, acc, feature_2d_lst):
    ''' 
    purpose: cope with single cable
    params: cable_index - cable index
            app_fund - approximate fundamental frequency
            fs - sampling frequency
            minute_interval - time interval in minutes
            acc - acceleration time history
            feature_2d_lst - [RMS_2d, PRS_2d, ...]
    return: None
    '''

    interval = int(minute_interval * 60 * fs)

    for i in range(acc.shape[0] // interval):

        try:
            error_flag, RMS, PRS, HCH, HCD, CCH, CCD, PRSM, HCHM, AHCHM = calc_VIV(
                acc[interval*i: interval*(i+1)], fs, app_fund, 1/6
            )
        except:
            print(f"cable_index: {cable_index}, time_interval_index: {i}")
            raise RuntimeError()

        if error_flag:  # 1, 0, 0, 0, 0, 0, 0, 0, 0, 0
            # error mark
            RMS = -1.

        for (f_2d, ft) in zip(feature_2d_lst, [
                RMS, PRS, HCH, HCD, CCH, CCD, PRSM, HCHM, AHCHM]):
            f_2d[cable_index].append(ft)

def find_all_csv(root):
    ''' 
    purpose: find all csv files in root 
    params: root - root path  
    return: feature_csv_lst - all feature csv files in root
    '''

    # feature_csv_lst = [RMS_csv_list, PRS_csv_list, ...]
    feature_csv_lst = [[] for _ in range(feature_num)]
    for file in os.listdir(root):
        file_path = os.path.join(root, file)
        if os.path.isdir(file_path):
            new_feature_csv_lst = find_all_csv(file_path)
            for i in range(feature_num):
                feature_csv_lst[i].extend(new_feature_csv_lst[i])
        else:
            file_name, ext = file.rsplit('.', 1)  # 2022-07-01 00_HCD.csv
            if ext != "csv":
                continue
            time, kind = file_name.rsplit('_', 1)
            for i in range(feature_num):
                if kind == features[i]:
                    feature_csv_lst[i].append(file_path)
                    break
    return feature_csv_lst

def _clear_csv(root):
    feature_csv_lst = find_all_csv(root)
    for csv_list in feature_csv_lst:
        for csv_file in csv_list:
            os.remove(csv_file)

def merge_one_feature(feature_csv, csv_dir):
    ''' 
    purpose: merge one feature
    params: feature_csv - all the csv files of one feature
            csv_dir - the directory of merged csv files
    return: None
    '''

    feature_2d = pd.DataFrame()
    for ft_file in sorted(feature_csv):
        feature_2d = pd.concat([feature_2d, pd.read_csv(ft_file, 
                header=None, dtype=np.float64).fillna(0)], axis=1)  # (36, 60)
    feature_type = feature_csv[0].rsplit('.', 1)[0].rsplit('_', 1)[-1]
    feature_2d.to_csv(os.path.join(csv_dir, f"{feature_type}.csv"), 
                      header=None, index=None)

def merge_RMS_PRS_etc(feature_csv_lst, csv_dir):
    ''' 
    purpose: merge all the features
    params: feature_csv_lst - all feature csv files in root
            csv_dir - the directory of merged csv files
    return: None
    '''

    pool = ProcessPoolExecutor()
    for i in range(feature_num):
        pool.submit(merge_one_feature, feature_csv_lst[i], csv_dir)
    pool.shutdown(True)
    feature_2d_lst = []
    for i in range(feature_num):
        feature_2d = pd.read_csv(os.path.join(csv_dir, f"{features[i]}.csv"), header=None)
        feature_2d_lst.append(feature_2d)
    # feature_2d_lst: (feature_num, 36, 60n) -> (feature_num, 60n, 36)
    feature_2d_lst = [pd.DataFrame(feature_2d.values.T) for feature_2d in feature_2d_lst]
    for i in range(feature_2d_lst[0].values.shape[1]):  # cable num
        cable_2d = pd.DataFrame()
        for j in range(feature_num):
            cable_2d = pd.concat([cable_2d, feature_2d_lst[j].iloc[:, [i]]], axis=1)
        cable_2d.to_csv(os.path.join(csv_dir, "ch%02d.csv" % (i+1)), header=features, index=None)
    for i in range(feature_num):
        os.remove(os.path.join(csv_dir, f"{features[i]}.csv"))

def get_index(csv_file):
    ''' 
    purpose: export the index correspondence between raw samples and effective samples
    params: csv_file - csv file that records all the (RMS, PRS, ...)
    return: None
    '''

    sample = pd.read_csv(csv_file).values
    right_index = (sample[:, 0] > 0)  # RMS > 0
    index_correspondence = np.zeros(right_index.sum(), dtype=np.int32)
    j: int = 0
    for i, v in enumerate(right_index):
        if v:
            index_correspondence[j] = i
            j += 1
    pd.DataFrame(index_correspondence.reshape(-1, 1)).to_csv(os.path.join(
        f"{csv_file.rsplit('.', 1)[0]}-index.csv"
    ), header=None, index=None)

def get_mat_index_table(mat_list):
    count_row = 0
    mat_index_table = np.zeros(len(mat_list), dtype=np.int32)
    for i, mat_file in enumerate(mat_list):
        try:
            acc_2d = loadmat(mat_file)["data"].astype(np.float64)  # (50*60*60=180000, 36)
        except:
            acc_2d = np.transpose(File(mat_file, 'r')["data"]).astype(np.float64)
        row_num = acc_2d.astype(np.float64).shape[0] // 3000
        count_row += row_num
        mat_index_table[i] = count_row
        if i % 50 == 0:
            print(i)
    return mat_index_table

def get_mat_minute(mat_index_table, mat_list, index):
    for (i, r) in enumerate(mat_index_table):
        if index < r:
            return (mat_list[i].rsplit('/', 1)[-1], 
                    index - (mat_index_table[i - 1] if i >= 1 else 0))

def main():
    root = "../../raw-data/Tongling"
    fs = 50
    minute_interval = 1
    # app_fund_arr = np.full((36,), 0.5)  # default app_fund: 0.5 Hz
    app_fund_arr = np.array([
        0.58, 0.59, 0.51, 0.58, 0.71,
        0.68, 0.60, 0.55, 0.51, 0.45, 
        0.54, 0.58, 0.69, 0.75, 0.56, 
        0.53, 0.61, 0.55, 0.70, 0.68,
        0.58, 0.54, 0.52, 0.51, 0.55, 
        0.59, 0.69, 0.73, 0.59, 0.69, 
        0.53, 0.57, 0.57, 0.53, 0.60, 
        0.56
    ])

    # _clear_csv(root) 
    # return

    # # step 1: process mat
    mat_list = sorted(find_all_mat(root))
    # cope_all_mat(mat_list, fs, app_fund_arr, minute_interval)

    # # step 2: merge
    # feature_csv_lst = find_all_csv(root)
    # merge_RMS_PRS_etc(feature_csv_lst, csv_dir="./data")

    # step 3: get index correspondence
    all_channel = ["ch%02d" % (i+1) for i in range(36)]
    for ch_name in all_channel:
        get_index(f"./data/{ch_name}.csv")

    # step 4: get mat index table
    mat_index_table = get_mat_index_table(mat_list)
    pd.DataFrame(mat_index_table.reshape(-1, 1)).to_csv(
            "./data/mat_index_table.csv", header=None, index=None)
    
    # # step 5: get mat and minute through index
    # mat_index_table = pd.read_csv("./mat_index_table.csv", header=None).values.flatten()
    # index = 239048
    # mat_file, minute = get_mat_minute(mat_index_table, mat_list, index)
    # print(f"mat: {mat_file}, minute: {minute}")

if __name__ == "__main__":
    t1 = time()
    main()
    t2 = time()
    print(t2- t1)
