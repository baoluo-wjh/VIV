import numpy as np
import matplotlib.pyplot as plt

plt.rc('font', family='Times New Roman', size=28)
plt.rc('text', usetex=True)

def plot_log():
    x = np.linspace(0, 1, 1001, endpoint=True)
    _, ax = plt.subplots(figsize=(10, 10))
    plt.scatter([0, 1], [0, 1], marker="*", s=512, c="tab:red")
    text_str = "$f \\left( x \\right) = 1 - \\frac{1}{t} \
            relu \\left[ -lg ( x+\\epsilon ) \\right]; \\epsilon = 10^{-t}$"
    plt.text(0.5, 0.15, text_str, color="tab:green", fontsize=28, 
             fontweight="bold", transform=ax.transAxes, va="center", ha="center")
    plt.axis("equal")
    plt.xlabel("$x$", size=48)
    plt.ylabel("$f \\left( x \\right)$", size=48)
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)
    for eps_t, cc in zip([2, 3, 4], ["tab:%s" % c for c in ["orange", "green", "blue"]]):
        x_modi = x.copy()
        eps = 0.1 ** eps_t
        x_modi = -np.log10(x_modi + eps)
        x_modi[x_modi < 0] = 0
        x_modi /= eps_t
        x_modi = 1 - x_modi
        y = x_modi
        plt.plot(x, y, linestyle='-', linewidth=4, color=cc, label="t=%d" % eps_t)
    plt.legend(loc="center right", fontsize=32)
    plt.savefig("./log.jpg", bbox_inches="tight", pad_inches=0.05)
    plt.close()

def plot_exp():
    x = np.linspace(0, 1, 1001, endpoint=True)
    _, ax = plt.subplots(figsize=(12, 10))
    alpha_lst = [1., 2., 3.]
    color_lst = ["tab:%s" % c for c in ["orange", "green", "blue"]]
    leg_1 = []
    leg_2 = []
    for alpha, cc in zip(alpha_lst, color_lst):
        y = np.exp(- alpha * x)
        fr = plt.plot(x, y, linestyle='-', linewidth=4, color=cc)
        leg_1.append(fr[0])
        plt.scatter([x[-1]], [y[-1]], marker="*", s=512, c=cc)
    for alpha, cc in zip(alpha_lst, color_lst):
        y = x * np.exp(- alpha * x)
        fwr = plt.plot(x, y, linestyle='--', linewidth=4, color=cc)
        leg_2.append(fwr[0])
    legend_1 = plt.legend(leg_1, ["$\\alpha=%.0f$" % alpha for alpha in alpha_lst], 
                          loc="upper right", fontsize=28, 
                          title="$f(r) = e ^ { - \\alpha r }$")
    legend_2 = plt.legend(leg_2, ["$\\alpha=%.0f$" % alpha for alpha in alpha_lst], 
                          loc="center right", bbox_to_anchor=(0.30, 0.37), fontsize=28, 
                          title="$f(r) = r \* e ^ { - \\alpha r }$")
    plt.gca().add_artist(legend_1)
    plt.scatter([0, 0], [0, 1], marker="*", s=512, c="tab:red")
    plt.axis("equal")
    plt.xlabel("$r$", size=48)
    plt.ylabel("$f(r)$", size=48)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)
    plt.savefig("./exp.jpg", bbox_inches="tight", pad_inches=0.05)
    plt.close()
    
if __name__ == "__main__":
    plot_log()
    plot_exp()
