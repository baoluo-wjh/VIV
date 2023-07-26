import os
import warnings
from time import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager
import numpy as np
from scipy.io import loadmat
from h5py import File
import matplotlib.pyplot as plt
import pandas as pd
from calc import calc_VIV
warnings.filterwarnings('error')
plt.rc('font', family='Times New Roman')
plt.rc('text', usetex = True)

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

def cope_all_mat(mat_list, fs, minute_interval=5):
    ''' 
    purpose: cope with all mat files by multiprocessing, 
             export RMS and HCD as csv files
    params: file_path - mat file's path
            fs - sampling frequency
            minute_interval - time interval in minutes
    return: None
    '''

    date_set = set()
    pool = ProcessPoolExecutor()  # max_workers=1
    for file_path in mat_list:  
        # "../HangZhou/south/2022-12-31 18-VIC.mat"
        date = file_path.rsplit('/', 1)[-1].split(' ')[0]
        # if date != "2022-01-04":
        #     continue
        # if date.split('-')[1] != "01":
        #     continue
        if date not in date_set:
            date_set.add(date)
            # print(f"{date} start")
        pool.submit(cope_mat, file_path, fs, minute_interval, False)
    pool.shutdown(True)

def cope_mat(file_path, fs, minute_interval=5, mp=True):
    ''' 
    purpose: cope with a mat file, export RMS and HCD as csv files
    params: file_path - mat file's path
            fs - sampling frequency
            minute_interval - time interval in minutes
    return: None
    '''

    mat_dir, mat_name = os.path.split(file_path)
    if os.path.exists(os.path.join(mat_dir, 
            f"{mat_name.rsplit('-', 1)[0]}_HCD.csv")): return
    
    try:
        acc_2d = loadmat(file_path)["data"].astype(np.float64)  # (50*60*60=180000, 36)
    except:
        acc_2d = np.transpose(File(file_path, 'r')["data"]).astype(np.float64)

    num_cable = acc_2d.shape[1]
    
    # concurrent container
    manager = Manager()
    con_lock = manager.RLock()
    RMS_2d = manager.list([manager.list() for _ in range(num_cable)])
    HCD_2d = manager.list([manager.list() for _ in range(num_cable)])

    if mp:  # concurrent process
        pool = ProcessPoolExecutor()  # max_workers=16
        for cable_index in range(num_cable):  # every cable in 60 min
            acc = acc_2d[:, cable_index]
            pool.submit(cope_cable, cable_index, fs, minute_interval, acc, 
                        con_lock, RMS_2d, HCD_2d)
        pool.shutdown(True)
    else:
        for cable_index in range(num_cable):  # every cable in 60 min
            acc = acc_2d[:, cable_index]
            try:
                cope_cable(cable_index, fs, minute_interval, acc, con_lock, RMS_2d, HCD_2d)
            except:
                print(f"mat_name: {mat_name}")

    # transform concurrent list to normal listmat_dir
    RMS_2d = [list(inner_lst) for inner_lst in RMS_2d]  # 36 * (60/5)
    HCD_2d = [list(inner_lst) for inner_lst in HCD_2d]

    # save RMS, HCD information
    pd.DataFrame(RMS_2d).to_csv(os.path.join(mat_dir, 
            f"{mat_name.rsplit('-', 1)[0]}_RMS.csv"), header=None, index=None)
    pd.DataFrame(HCD_2d).to_csv(os.path.join(mat_dir, 
            f"{mat_name.rsplit('-', 1)[0]}_HCD.csv"), header=None, index=None)

def cope_cable(cable_index, fs, minute_interval, acc, con_lock, RMS_2d, HCD_2d):
    ''' 
    purpose: cope with single cable
    params: cable_index - cable index
            fs - sampling frequency
            minute_interval - time interval in minutes
            acc - acceleration time history
            con_lock - concurrent lock
            RMS_2d - shared RMS
            HCD_2d - shared HCD
    return: None
    '''

    interval = int(minute_interval * 60 * fs)

    for i in range(acc.shape[0] // interval):

        try:
            error_flag, RMS, HCD = calc_VIV(acc[interval*i: interval*(i+1)], fs)
        except:
            print(f"cable_index: {cable_index}, time_interval_index: {i}")
            raise RuntimeError()

        if error_flag:
            # error mark
            HCD += 10

        con_lock.acquire()
        RMS_2d[cable_index].append(RMS) 
        HCD_2d[cable_index].append(HCD)
        con_lock.release()

def find_all_csv(root):
    ''' 
    purpose: find all csv files in root 
    params: root - root path  
    return: RMS_csv_list - all RMS csv files in root
            HCD_csv_list - all HCD csv files in root
    '''

    RMS_csv_list = []
    HCD_csv_list = []
    for file in os.listdir(root):
        file_path = os.path.join(root, file)
        if os.path.isdir(file_path):
            new_RMS_csv, new_HCD_csv = find_all_csv(file_path)
            RMS_csv_list.extend(new_RMS_csv)
            HCD_csv_list.extend(new_HCD_csv)
        else:
            file_name, ext = file.rsplit('.', 1)  # 2022-07-01 00_HCD.csv
            if ext != "csv":
                continue
            time, kind = file_name.rsplit('_', 1)
            if kind == "RMS":
                RMS_csv_list.append(file_path)
            elif kind == "HCD":
                HCD_csv_list.append(file_path)
    return RMS_csv_list, HCD_csv_list

def merge_RMS_HCD(RMS_csv_list, HCD_csv_list, csv_dir):
    ''' 
    purpose: find all csv files in root 
    params: RMS_csv_list - all RMS csv files in root
            HCD_csv_list - all HCD csv files in root
            csv_dir - the directory of merged csv files
    return: None
    '''

    for file in os.listdir(csv_dir):
        if file.rsplit('.', 1)[-1] == "csv":
            return

    date_set = set()

    RMS_2d = pd.DataFrame()
    HCD_2d = pd.DataFrame()
    for (RMS_file, HCD_file) in zip(RMS_csv_list, HCD_csv_list):
        RMS_2d = pd.concat([RMS_2d, pd.read_csv(RMS_file,  # (36*12)
                header=None, dtype=np.float64).fillna(0)], axis=1) 
        HCD_2d = pd.concat([HCD_2d, pd.read_csv(HCD_file, 
                header=None, dtype=np.float64).fillna(0)], axis=1) 
        date = HCD_file.rsplit('/', 1)[-1].split(' ', 1)[0]
        if date not in date_set:
            date_set.add(date)
            print(f"{date} merged")

    RMS_2d = pd.DataFrame(RMS_2d.values.T)  # (36 * 12n) -> (12n * 36)
    HCD_2d = pd.DataFrame(HCD_2d.values.T)
    for i in range(RMS_2d.values.shape[1]):
        pd.concat([RMS_2d[RMS_2d.columns[i]], HCD_2d[HCD_2d.columns[i]]], 
                  axis=1).to_csv(os.path.join(csv_dir, "ch%02d.csv" % (i+1)), 
                                 header=["RMS", "HCD"], index=None)

def main():
    root = "../HangZhou/north"  # north
    fs = 50
    minute_interval = 1
    mat_list = sorted(find_all_mat(root))  
    # date_set = set()
    # for mat_file in mat_list:
    #     # "../HangZhou/south/2022-12-31 18-VIC.mat"
    #     date = mat_file.rsplit('/', 1)[-1].split(' ')[0]  
    #     if date not in date_set:
    #         date_set.add(date)
    #         # print(f"{date} start")
    #     if date != "2022-01-04":
    #         continue
    #     cope_mat(mat_file, fs, minute_interval)
    cope_all_mat(mat_list, fs, minute_interval)
    RMS_csv_list, HCD_csv_list = find_all_csv(root)
    RMS_csv_list = sorted(RMS_csv_list)
    HCD_csv_list = sorted(HCD_csv_list)
    # print(list(zip(RMS_csv_list, HCD_csv_list)))
    csv_dir = "./src"
    merge_RMS_HCD(RMS_csv_list, HCD_csv_list, csv_dir)

if __name__ == "__main__":
    t1 = time()
    main()
    t2 = time()
    print(t2- t1)
