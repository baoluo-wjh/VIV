import pandas as pd
from scipy.io import loadmat
from calc import calc_VIV

def identify_VIV(acc, fs, boundary):
    ''' 
    purpose: identify VIV
    params: acc - time history series of acceleration
            fs - sampling frequency
            boundry - (a, b, c)
    return: True if VIV happens else False
    '''

    error_flag, RMS, HCD = calc_VIV(acc, fs)
    if error_flag:
        return False
    # X = np.array([RMS, HCD]).reshape(-1, 2)
    a, b, c = boundary
    return True if HCD > -a/b*RMS - c/b else False

def main():
    boundry = tuple(pd.read_csv("./src/ch02-abc.csv", header=None).values.flatten())
    fs = 50
    a = loadmat("../铜陵桥数据22.7-23.2/2022-07-01/2022-07-01 00-VIC.mat")\
            ["data"][:, 1]
    acc = a[:a.shape[0] // 12]
    flag = identify_VIV(acc, fs, boundry)
    print(flag)

if __name__ == "__main__":
    main()
