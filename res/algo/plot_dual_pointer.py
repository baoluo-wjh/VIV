import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
plt.rc('font', family='Times New Roman')
plt.rc('text', usetex = True)

def filter_freq(f, specs, threshold):
    wave_filter = (f > threshold)
    specs = specs * wave_filter
    specs = specs / specs.max() 
    return f, specs

def find_peak(arr, interval):
    ''' 在序列 arr 中寻找间隔 interval 内的最大元素，最大元素不应小于 min_peak '''
    peaks = []
    N = len(arr)
    mark = np.ones_like(arr, dtype='bool')
    for i in range(N):
        if not mark[i]:
            continue
        for j in range(1, interval//2):
            if i + j >= N:
                break
            if arr[i + j] < arr[i]:
                mark[i + j] = False
            else:
                mark[i] = False
                break
        if not mark[i]:
            continue
        for j in range(-interval // 2, 0):
            if i + j >= 0 and arr[i + j] > arr[i]:
                mark[i] = False
                break
        if mark[i]: 
            peaks.append(i)
    return peaks

def get_peak(Fs, f, specs, freq_width):
    '''
    specs 是已经归一化了的实数振幅半谱
    freq_width 和 min_peak 是向量，用户可对各索进行精准把控
    '''
    total_eigen_freqs = []
    interval = (freq_width / Fs).astype(int)
    peaks = find_peak(specs, interval)  # 筛法选峰
    # 经实验验证，若 peaks==[] 则 total_eigen_freqs 实际压入了一个 []
    total_eigen_freqs.append(list(zip(f[peaks], specs[peaks])))
    return total_eigen_freqs

def predict_order(total_eigen_freqs):
    '''
    total_eigen_freqs 是列表，其元素也是列表，里面是元组，元组包含特征频率及其幅值
    freq_th 是特征频率的理论值，可由用户提供，是列表
    本函数预测并返回各索各特征频率的阶次，注意这里的阶次是全的，因为没有设置阈值 max_f
    另外，本函数还会根据 order_lst 反过来修正 total_eigen_freqs
    '''
    order_lst = []
    num = len(total_eigen_freqs)
    # 依次处理每一根索
    for i in range(num):
        freq_lst   = []
        A_lst      = []
        for (freq, A) in total_eigen_freqs[i]:
            freq_lst.append(freq)
            A_lst.append(A)
        # 没有识别出特征频率，手动进行异常处理，采用静默失败
        if len(freq_lst) == 0:
            order_lst.append([])
            continue
        # 实践证明，当识别出的特征频率的数量小于 3 时，算法的鲁棒性非常差，
        # 所以当做异常来处理，采用静默失败
        if len(freq_lst) < 3:
            order_lst.append([])
            # 将之前识别的特征频率重置为空
            total_eigen_freqs[i] = []
            continue
        # 求特征频率差分
        diff_freq = []
        amplitude_weight = []
        # 邹总说一阶不能要，故舍去第一个差分
        for k in range(2, len(freq_lst)):
            diff_freq.append(freq_lst[k]-freq_lst[k-1])
            # 采用 min 函数，要求作差分的两个频率的幅值都较大
            # 这是一种较严格的要求，内涵上相当于逻辑与：&&
            amplitude_weight.append( min(A_lst[k], A_lst[k-1]) )
        # 投票算法，count 是字典，键为整数，值为小数票
        count = dict()
        # 乘子越大哈希值越发散，噪声点的投票越离散，从而越容易排除噪声干扰；
        # 但是过大会导致正确差分的投票也无法集中，从而无法得到正确的基频
        multiplier = 200
        # 定义哈希函数
        hash_func = lambda x: multiplier*x
        # 投票
        for j in range(len(diff_freq)):
            hash_val = hash_func(diff_freq[j])
            low_int  = int(hash_val)
            high_int = low_int + 1
            # 给附近的两个整数都投小数票，票数由相近度和振幅决定
            A_weight = amplitude_weight[j]
            count[low_int] =count.get(low_int, 0)+(high_int-hash_val)*A_weight
            count[high_int]=count.get(high_int,0)+(hash_val-low_int )*A_weight
        # 对字典依键进行从小到大的排序，键（哈希值）小的排前面
        count = dict(sorted(count.items()))
        # 统计得票最多的候选人，相同票数下优先选值小的候选人
        base_freq_map = 0
        ticket        = 0
        for (eigen, tic) in count.items():
            if tic > ticket:
                ticket = tic
                base_freq_map = eigen
        # 如果投票结果为 0 则基频识别无效，采用静默失败
        if base_freq_map == 0:
            order_lst.append([])
            total_eigen_freqs[i] = []
            continue     
        # 还原基频。采用以贡献票数为权的加权平均数
        base_freq    = 0
        total_ticket = 0
        for j in range(len(diff_freq)):
            delta_val = abs( hash_func(diff_freq[j])-base_freq_map )
            if delta_val <= 1:
                # 总权重 == 贡献票数 == 相近度*振幅权
                weight = (1-delta_val) * amplitude_weight[j]
                base_freq += weight * diff_freq[j]
                total_ticket += weight
        base_freq /= total_ticket
        # 如果 base_freq 明显不是基频，则静默失败
        if base_freq / freq_lst[0] > 1.5:
            order_lst.append(([]))
            total_eigen_freqs[i] = []
            continue
        # 至此，已经估算出基频
        # 接下来推算各峰值点的阶次
        order = []
        # selected_eigen_index 用于记录 total_eigen_freqs[i] 中有效的特征频率的索引
        selected_eigen_index = []
        # 计算第一个有效的特征频率的阶次
        for init in range(len(freq_lst)):
            val = freq_lst[init] / base_freq
            n0 = int(round(val))
            # val 几乎是整数
            if abs(val - n0) <= 0.10:
                order.append(n0)
                selected_eigen_index.append(init)
                break
        # 差分累加算法，双指针补上剩下的阶次
        former = init
        for later in range(init+1, len(freq_lst)):
            delta_val = (freq_lst[later]-freq_lst[former]) / base_freq
            delta_n = int(round(delta_val))
            # delta_val 几乎是整数
            if abs(delta_val-delta_n) <= 0.10:
                n0 += delta_n
                order.append(n0)
                selected_eigen_index.append(later)
                former = later
        order_lst.append(order)
        # 修正之前识别的特征频率
        modified_eigen_freqs = []
        for j in selected_eigen_index:
            modified_eigen_freqs.append(total_eigen_freqs[i][j])
        total_eigen_freqs[i] = modified_eigen_freqs
    return order_lst

def plot_dual_pointer(f, specs, max_f, total_eigen_freqs):
    fig, axes = plt.subplots(6, 1, figsize=(20, 15))
    for i, ax in enumerate(axes):
        mask = (f < max_f)
        ax.plot(f[mask], specs[mask], alpha=0.5, linewidth=6)
        ax.set_yticks([])
        ax.set_xlim([0, max_f])
        ax.set_ylim([0, 1.50])
        ax.tick_params(labelsize=40)
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
        # ax.hlines(0, 0, max_f, colors='r', linestyles="dashdot", lw=4)
        xs = []
        for k, (freq, A) in enumerate(total_eigen_freqs[0]):
            if freq > max_f:
                break
            # ax.vlines(freq, 0, A, color='r', lw=5)
            xs.append(freq)
            if k == i:
                arrow_poly = mpatches.Polygon([
                    [freq-0.01, A+0.2],
                    [freq, A+0.1],
                    [freq+0.01, A+0.2]
                ], color='g')
                arrow_rect = mpatches.Rectangle((freq-0.005, A+0.2), 0.01, 0.3, color='g')
                ax.add_patch(arrow_poly)
                ax.add_patch(arrow_rect)
                ax.text(freq+0.03, A+0.20, f"$i={k}$", color='g', fontsize=40)
            if k == i and i != 0:
                x = total_eigen_freqs[0][k-1][0]
                y = total_eigen_freqs[0][k-1][1]
                arrow_poly = mpatches.Polygon([
                    [x-0.01, y+0.2],
                    [x, y+0.1],
                    [x+0.01, y+0.2]
                ], color='purple')
                arrow_rect = mpatches.Rectangle((x-0.005, y+0.2), 0.01, 
                        0.3, color='purple')
                ax.add_patch(arrow_poly)
                ax.add_patch(arrow_rect)
                ax.text(x+0.03, y+0.23, f"$j={k-1}$", color='purple', fontsize=40)
        if i == 5:
            ax.set_xticks(xs)
        else:
            ax.set_xticks([])
    plt.savefig("WHV-spectrum.pdf", bbox_inches='tight')
    plt.close()

def save_total_eigen_freqs(max_f, total_eigen_freqs):
    freq_lst = []
    amp_lst = []
    for (freq, amp) in total_eigen_freqs[0]:
        if freq > max_f:
            break
        freq_lst.append(freq)
        amp_lst.append(amp)
    diff_lst = []
    amp_wei_lst = []
    for i in range(len(freq_lst) - 1):
        diff_lst.append(freq_lst[i+1]-freq_lst[i])
        amp_wei_lst.append(min(amp_lst[i+1], amp_lst[i]))
    hash_lst = []
    low_int = []
    high_int = []
    low_pro_wei = []
    high_pro_wei = []
    for val in diff_lst:
        hash_val = 200 * val
        hash_lst.append(hash_val)
        low = int(hash_val)
        high = low + 1
        low_int.append(low)
        high_int.append(high)
        low_pro_wei.append(high - hash_val)
        high_pro_wei.append(hash_val - low)
    hash_lst.append(0)
    diff_lst.append(0) 
    amp_wei_lst.append(0) 
    low_int.append(0) 
    low_pro_wei.append(0) 
    high_int.append(0) 
    high_pro_wei.append(0) 
    pd.DataFrame(np.array([freq_lst, amp_lst, diff_lst, amp_wei_lst, hash_lst, low_int, 
            low_pro_wei, high_int, high_pro_wei]).T).to_csv("WHV-process.csv", 
            float_format='%.4f', header=None, index=None)

def main():
    max_f = 2.5
    threshold = 0.2
    freq_width = 0.33 * 1.5
    f = pd.read_csv("../1-1-ini/f.csv", header=None).values[:, 0]
    specs = pd.read_csv("../1-1-ini/specs.csv", header=None).values[3] 
    Fs = f[1]
    mask = (f > 1.3233)
    f = f[mask]
    specs = specs[mask]
    f -= 1.3233
    f, specs = filter_freq(f, specs, threshold)
    total_eigen_freqs = get_peak(Fs, f, specs, freq_width)
    predict_order(total_eigen_freqs)
    plot_dual_pointer(f, specs, max_f, total_eigen_freqs)
    save_total_eigen_freqs(max_f, total_eigen_freqs)

if __name__ == "__main__":
    main()
