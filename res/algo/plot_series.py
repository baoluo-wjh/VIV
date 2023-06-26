import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
plt.rc('font', family='Times New Roman')
plt.rc('text', usetex=True)

arr = np.random.randint(low=0, high=3, size=20, dtype=np.int32)
arr[0] = 15
arr[3] = 8  # big noise
arr[6] = 13
arr[12] = 20
arr[15] = 6  # big noise
arr[18] = 17

def plot_process():
    maxima = [0, 6, 12, 18]
    indexes = [0, 5, 6, 11, 12, 17, 18]
    stars = [
        [0],
        [0],
        [0, 6],
        [0, 6],
        [0, 6, 12],
        [0, 6, 12],
        [0, 6, 12, 18]
    ]
    crosses = [
        [1, 2, 3, 4],
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5, 7, 8, 9, 10],
        [1, 2, 3, 4, 5, 7, 8, 9, 10, 11],
        [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 13, 14, 15, 16],
        [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17],
        [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 19]
    ]
    fig, axes = plt.subplots(7, 1, figsize=(20, 15))
    for i, ax in enumerate(axes):
        ax.plot(range(20), arr, alpha=0.5)
        if i == 6:
            ax.set_xticks(range(0, 22, 2))
        else:
            ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim([-5, 23])
        # ax.set_xlim([-0.5, 19.5])
        ax.set_ylim([-4, 23])
        plt.setp(ax.get_xticklabels(), fontsize=36)
        ax.scatter(stars[i], arr[stars[i]], color='y', marker='*', s=512)
        ax.scatter(crosses[i], arr[crosses[i]], color='b', marker='x', s=256)
        ax.text(0.04, 0.3, f"Current\nIndex: {indexes[i]}", color='g', 
                fontsize=36, transform=ax.transAxes)
        mid = indexes[i]
        left = mid - 4.2
        rect = mpatches.Rectangle((left, -2), 8.4, 23, color="r", 
                linewidth=3, fill=False)
        ax.add_patch(rect)
        if mid in maxima:
            arrow_poly = mpatches.Polygon([
                [mid-0.1, arr[mid]-5],
                [mid, arr[mid]-2],
                [mid+0.1, arr[mid]-5]
            ], color='r')
            arrow_rect = mpatches.Rectangle((mid-0.05, arr[mid]-11), 0.1, 6, color='r')
        else:
            arrow_poly = mpatches.Polygon([
                [mid-0.1, arr[mid]+5],
                [mid, arr[mid]+2],
                [mid+0.1, arr[mid]+5]
            ], color='r')
            arrow_rect = mpatches.Rectangle((mid-0.05, arr[mid]+5), 0.1, 6, color='r')
        ax.add_patch(arrow_poly)
        ax.add_patch(arrow_rect)
    plt.savefig("FSM-process.pdf", bbox_inches='tight')
    plt.close()

def plot_cmp():
    maxima = [0, 6, 12, 18]
    ws = [5, 9, 13]
    maxima_lst = [
        [0, 3, 6, 12, 15, 18],
        [0, 6, 12, 18],
        [0, 12]
    ]
    arrow_lst = [
        [6, 12, 15],
        [6, 12, 15],
        [6, 12, 15]
    ]
    window_lst = [12, 12, 12]
    fig, axes = plt.subplots(3, 1, figsize=(20, 10))
    for i, ax in enumerate(axes):
        ax.plot(range(20), arr, alpha=0.5)
        if i == 2:
            ax.set_xticks(range(0, 22, 2))
        else:
            ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim([-1, 20])
        ax.set_ylim([-4, 25])
        plt.setp(ax.get_xticklabels(), fontsize=42)
        star = maxima_lst[i]
        cross = [j for j in range(20) if j not in maxima_lst[i]]
        ax.scatter(star, arr[star], color='y', marker='*', s=512)
        ax.scatter(cross, arr[cross], color='b', marker='x', s=256)
        ax.text(0.06, 0.75, "$\\it{ws}$ = %d" % ws[i], 
                color='g', fontsize=42, transform=ax.transAxes)
        rect = mpatches.Rectangle((window_lst[i] + 0.3 - ws[i] / 2, -2), 
                ws[i] - 0.6, 25, color="r", linewidth=3, fill=False)
        ax.add_patch(rect)
        for mid in arrow_lst[i]:
            if mid in maxima:
                arrow_poly = mpatches.Polygon([
                    [mid-0.1, arr[mid]-5],
                    [mid, arr[mid]-2],
                    [mid+0.1,arr[mid]-5]
                ], color='r')
                arrow_rect = mpatches.Rectangle((mid-0.05, arr[mid]-9), 
                        0.1, 4, color='r')
                ax.text(mid+0.1, arr[mid]-9, "%d" % mid, color='r', fontsize=32)
            else:
                arrow_poly = mpatches.Polygon([
                    [mid-0.1, arr[mid]+5],
                    [mid, arr[mid]+2],
                    [mid+0.1,arr[mid]+5]
                ], color='r')
                arrow_rect = mpatches.Rectangle((mid-0.05, arr[mid]+5), 
                        0.1, 4, color='r')
                ax.text(mid+0.1, arr[mid]+5, "%d" % mid, color='r', fontsize=32)
            ax.add_patch(arrow_poly)
            ax.add_patch(arrow_rect)
    plt.savefig("FSM-numToCmp.pdf", bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    plot_process()
    plot_cmp()
