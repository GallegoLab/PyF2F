#!/bin/python3
import sys
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
sns.set_style('whitegrid')
sns.set(rc={'figure.figsize': (15, 35)})


def plot_comparison(data, save_path, title):
    """
    Point plot + error bars for each mu estimation
    """
    fig, ax = plt.subplots(figsize=(20, 8))
    # ax.yaxis.grid(True, color="black")
    # ax.grid(True, color="grey", alpha=0.5)
    ax.set_facecolor('w')
    ax.set_title(title, fontdict={"fontsize": 20}, loc="center", pad=15.0)
    ax.errorbar(x=[x * 0.1 for x in list(range(len(data)))], y=data.mu_cell, yerr=data.serr_mu_cell,
                fmt="^", c="orange", capsize=5, markersize=7,  label="CELL")
    ax.errorbar(x=[(x + 0.3) * 0.1 for x in list(range(len(data)))], y=data.mu_py, yerr=data.serr_mu_py,
                fmt="x", c="blue", capsize=5, markersize=7, label="Python")
    ax.set_xticks([x * 0.1 for x in list(range(len(data)))])
    ax.set_xticklabels(labels=data.restraint, rotation=40, ha="right")
    ax.set_ylabel(ylabel="d (nm)", fontdict={"fontsize": 20}, labelpad=10.0)
    ax.set_xlabel(xlabel="prey-GFP - anchor-RFP distances", fontdict={"fontsize": 20}, labelpad=10.0)
    plt.legend(fontsize=20)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    # Color xticks and labels based on similar numbers
    non_sel_labels = data.loc[data.selected == 'non-sel', 'restraint'].to_list()
    # change_ax_colors(ax, non_sel_labels)
    plt.tight_layout()
    # plt.show()
    plt.savefig(save_path, dpi=100)


def change_ax_colors(ax, non_sel_labels):
    """
    Change x label color based on condition
    (those that are not within the same params)
    """
    labels = ax.get_xticklabels()
    ticks = ax.get_xticks()
    for label, tick in zip(labels, ticks):
        if label.get_text() in non_sel_labels:
            label.set_color('r')


def plot_stats_distributions(data, save_path, title):
    """
    Compare the two samples (N-ter or C-ter) and accept or reject the hypothesis
    that the two samples comes from the same population. We know that both samples
    are extracted from the same set of images, so we should not reject the null
    hypothesis with a two-sample Kolmogorov-Smirnov test for goodness of fit.

    """
    # Comput KS test
    result = stats.ks_2samp(data.mu_cell.to_numpy(), data.mu_py.to_numpy())
    # Compute and plot linear fit
    g = sns.lmplot(data=data, x='mu_cell', y='mu_py', ci=95).set(title=f'{title}\nlm fit model and KS stat')
    for ax in g.axes.flat:
        ax.text(1.2, 0.85, f'KS test: {round(result[0], 3)}\n'
                           f'pvalue = {round(result[1], 3)}', fontsize=9)
    g.savefig(save_path, dpi=100)


if __name__ == '__main__':
    path = "../Cell_PyF2F_comparison/"

    # Data for N and C ter differences
    data_n_ter = pd.read_csv(f"{path}N_terminal.csv")
    data_c_ter = pd.read_csv(f"{path}C_terminal.csv")
    # data_Sec2 = pd.read_csv(f"{path}Sec2.csv")

    data_n_ter.loc[:, "restraint"] = [f"{x[0]}-{x[1]}" for x in data_n_ter.to_numpy()]
    data_c_ter.loc[:, "restraint"] = [f"{x[0]}-{x[1]}" for x in data_c_ter.to_numpy()]
    # data_Sec2.loc[:, "restraint"] = [f"{x[0]}-{x[1]}" for x in data_Sec2.to_numpy()]

    # Plot differences in barplot
    plot_comparison(data_n_ter, f'{path}comparison_N.png', title="N-terminal")
    plot_comparison(data_c_ter, f'{path}comparison_C.png', title="C-terminal")
    # plot_comparison(data_Sec2, f'{path}comparison_Sec2.png', title="Sec2")

    print("\nDONE!\n")
    sys.exit(0)
