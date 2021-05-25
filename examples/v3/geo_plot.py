import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import geom

def test():
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsfonts,amsmath}')
    plt.rcParams['font.family'] = ['serif']
    plt.rcParams['axes.titlepad'] = 0.02
    # plt.rcParams['xtick.major.pad'] = 0.5
    plt.rcParams['ytick.major.pad'] = 0.05
    plt.rcParams['axes.labelpad'] = 0.8
    plt.rcParams['axes.linewidth'] = 0.8
    fontsize=10
    labelsize=7

    # plt.rcParams.update({'font.size': fontsize})


    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3.5, 2.2))
    p=0.2
    x = geom.rvs(p, size=2000)
    # x = np.random.randn(200)
    colors = plt.cm.viridis(np.linspace(0.3, 1, 2))[-2:]
    kwargs = {'cumulative': True}

    n, bins, patches = ax.hist(x, cumulative=True, density=True, bins=50, alpha=1, color=colors[0])

    theoretical_y = []
    for i  in bins:
        theoretical_y.append(geom.cdf(i, p))

    ax.plot(bins, theoretical_y, 'k--', linewidth=1.5, label='Theoretical')

    # set the grid on
    # ax.grid('on')

    ax.set_title(r"$X \sim Ge(p=0.2)$", fontsize=fontsize)
    ax.set_xlabel(r"Intrusion start time $t$", fontsize=labelsize)
    ax.set_ylabel(r"$CDF_X(t)$", fontsize=labelsize)
    ax.set_xlim(1, 29)

    # tweak the axis labels
    xlab = ax.xaxis.get_label()
    ylab = ax.yaxis.get_label()

    xlab.set_size(11.5)
    ylab.set_size(11.5)

    # change the color of the top and right spines to opaque gray
    ax.spines['right'].set_color((.8, .8, .8))
    ax.spines['top'].set_color((.8, .8, .8))

    fig.tight_layout()
    # plt.show()
    # plt.subplots_adjust(wspace=0, hspace=0)
    fig.savefig("geo_plot" + ".png", format="png", dpi=600)
    fig.savefig("geo_plot" + ".pdf", format='pdf', dpi=600, bbox_inches='tight', transparent=True)
    # sns.distplot(x, hist_kws=kwargs, kde_kws=kwargs)

if __name__ == '__main__':
    test()