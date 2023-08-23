import numpy as np
import matplotlib.pyplot as plt

plt.rc('font', family='Times New Roman')
plt.rc('text', usetex = True)

def main():
    x = np.linspace(0, 1, 1001, endpoint=True)
    x_modi = x.copy()
    eps_t = 2
    eps = 0.1 ** eps_t
    x_modi = -np.log10(x_modi + eps)
    x_modi[x_modi < 0] = 0
    x_modi /= eps_t
    x_modi = 1 - x_modi
    y = x_modi
    # plot pdf
    _, ax = plt.subplots(figsize=(10, 10))
    plt.plot(x, y, linestyle='-', linewidth=4, color="tab:blue")
    plt.scatter([0, 1], [0, 1], marker="*", s=512, c="tab:orange")
    text_str = "$f \\left( x \\right) = 1 - \\frac{1}{t} \
            relu \\left[ -lg \\left( x+\\epsilon \\right) \\right]$ \n"
    text_str += "$t = %d; \\epsilon = 10^{-t}$" % (eps_t)
    plt.text(0.5, 0.5, text_str, color="tab:green", fontsize=36, 
             fontweight="bold", transform=ax.transAxes, va="center", ha="center")
    plt.axis("equal")
    plt.xlabel("$x$", size=48)
    plt.ylabel("$f \\left( x \\right)$", size=48)
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)
    plt.savefig("./log.jpg", bbox_inches="tight", pad_inches=0.05)
    plt.close()
    
main()
