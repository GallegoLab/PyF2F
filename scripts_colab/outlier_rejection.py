#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Python functions to run outlier rejection of measured distances.
It has the  functions to estimate the 'mu' and 'sigma' parameters
through a maximum likelihood estimate (MLE) given the probability
density function calculated by Churchman et al., 2005.
"""
import sys
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from rnd import pdf, cdf
from lle import bootstrap, LL
from skimage.io import imread


def custom_outlier_read_csv(file):
    """
    Function to read multiple csv (input data)
    """
    cols = ["ID", "x", "y", "img", "distances"]
    df = pd.read_csv(file, usecols=cols, sep="\t")
    return df


def save_html_kde(path_to_save, channel_image, sub_df, img_num, channel_name):
    """
    Method to save image with scattered spots as html
    Parameters
    ----------
    :param channel_image: ndimage corresponding to channel W1 or W2
    :param path_to_save: path to save figure in html
    :param sub_df: sub-dataframe to work with
    :param img_num: image number
    :param channel_name: channel name ("W1" or "W2")

    """
    if not os.path.exists(path_to_save + "outlier/"):
        os.mkdir(path_to_save + "outlier/")

    selected = sub_df[sub_df["selected"] == "sel"]
    non_selected = sub_df[sub_df["selected"] == "non-sel"]
    percent_sel = round(len(selected) * 100 / (len(selected) + len(non_selected)), 3)
    foo_note = "<br>Number of Selected spots: {} / {} (Percentage = {} %)<br><br>".format(len(selected),
                                                                                          len(selected) + len(
                                                                                              non_selected),
                                                                                          percent_sel)

    # Create figure with lines to the closest contour and closest neighbour
    fig_label_cont = px.imshow(channel_image, color_continuous_scale='gray',
                               title="<b>ImageMD {} {} - Outlier selected</b><br>{}".format(img_num,
                                                                                            channel_name, foo_note))
    fig_label_cont.update_layout(coloraxis_showscale=False)  # to hide color bar

    # Plot spots with custom hover information
    fig_label_cont.add_scatter(x=selected["y"], y=selected["x"],
                               mode="markers",
                               marker=dict(color="green", size=7),
                               name="W1",
                               hovertemplate=
                               '<b>x: %{x: }</b><br>'
                               '<b>y: %{y: } <b><br>')

    fig_label_cont.add_scatter(x=non_selected["y"], y=non_selected["x"],
                               mode="markers",
                               marker=dict(color="red", size=7),
                               name="W2",
                               hovertemplate=
                               '<b>x: %{x: }</b><br>'
                               '<b>y: %{y: } <b><br>')
    # Save figure in output directory
    fig_label_cont.write_html(path_to_save + "outlier/" + "image_{}_{}.html".format(img_num, channel_name))


def plot_outlier_rejection(sel_distribution, c, mu, sigma, i_max, n, sh_scores, dataset_name, results_dir,
                           figures_dir, bin_size):
    """
    Plot distance distributio BEFORE and AFTER outlier
    rejection

    Parameters
    ----------
    :param sel_distribution:
    :param c
    :param mu
    :param sigma
    :param i_max
    :param n
    :param sh_scores
    :param dataset_name
    :param results_dir
    :param figures_dir
    :param bin_size: size of bins for plots

    """
    xlim = 180
    initial_distances = np.loadtxt(results_dir + "distances_after_warping.csv")
    np.savetxt(results_dir + "final_distances.csv", sel_distribution, delimiter="\t")

    # PLOT NEW DISTANCE DISTRIBUTION
    fig, ax = plt.subplots(figsize=(25, 20))
    sns.set(font_scale=3)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.despine()
    ax.set_title("Distances after OUTLIER rejection\n\n"
                 "mean initial = {} nm; stdev initial = {} nm; n = {}\n"
                 "mean final = {} nm; stdev final = {} nm; "
                 "n = {} \n".format(np.around(np.mean(initial_distances), 2),
                                    np.around(np.std(initial_distances), 2),
                                    len(initial_distances),
                                    np.around(np.mean(sel_distribution), 2),
                                    np.around(np.std(sel_distribution), 2),
                                    len(sel_distribution)),
                 fontweight="bold", size=25)
    sns.histplot(data=initial_distances, kde=False, color="tab:grey", binwidth=bin_size, ax=ax, fill=True,
                 stat="density")
    sns.histplot(data=sel_distribution, kde=False, ax=ax, color="mediumturquoise", binwidth=bin_size, fill=True,
                 stat="density")
    ax.set_xlabel("Distances (nm) ", fontsize=30, labelpad=30)
    ax.set_ylabel("Density ", fontsize=30, labelpad=30)
    ax.tick_params(axis='x', labelsize=30)
    ax.tick_params(axis='y', labelsize=30)
    ax.set_xlim([0, xlim])
    # ax.axvline(x=np.mean(initial_distances), color='sandybrown', ls='--', lw=2.5, alpha=0.8)
    # ax.axvline(x=np.mean(sel_distribution), color='cornflowerblue', ls='--', lw=2.5, alpha=0.8)
    ax.plot(c[0], pdf(c[0], mu[i_max][0], sigma[i_max][0]), color='black', linewidth=8,
            label='$\mu=$' + str(round(mu[i_max][0], 2)) + '$\pm$' + str(round(mu[i_max][1], 2))
                  + 'nm, $\sigma=$' +
                  str(round(sigma[i_max][0], 2)) + '$\pm$' + str(round(sigma[i_max][1], 2)) + 'nm'
                  + ", n=" + str(n))
    plt.savefig(figures_dir + "final_distance_distribution.png")
    plt.clf()

    # OUTPUT SUMMARY: PLOT FINAL DISTANCE DISTRIBUTION WITH FIT AND SCORES
    # ax1 Distribution with fit
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(40, 15))
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.despine()
    sns.histplot(x=sel_distribution, kde=False, ax=ax1, stat="density", legend=True, color='mediumturquoise',
                 binwidth=bin_size, edgecolor='black')
    ax1.plot(c[0], pdf(c[0], mu[i_max][0], sigma[i_max][0]), color='black', linewidth=8,
             label=
             '$\mu=$' + str(round(mu[i_max][0], 2)) + '$\pm$' + str(round(mu[i_max][1], 2))
             + 'nm, $\sigma=$' +
             str(round(sigma[i_max][0], 2)) + '$\pm$' + str(round(sigma[i_max][1], 2)) + 'nm'
             + ", n=" + str(n))
    # ax1.axvline(x=mu[i_max][0], color='black', ls='--', lw=3, alpha=0.3)
    ax1.set_xlabel('distance (nm)', fontsize=30)
    ax1.set_ylabel('Density', fontsize=30)
    ax1.set_xlim([0, max(sel_distribution)])  # max(sel_distribution)
    legend = ax1.legend(loc='upper right', shadow=True, fontsize=30)
    legend.get_frame().set_facecolor('C0')
    ax1.tick_params(axis='x', labelsize=25)
    ax1.tick_params(axis='y', labelsize=25)

    # Shannon Scoring Plot
    shannon_data = list(zip([mu[i][0] for i in range(len(sh_scores))], sh_scores))
    dist_list, shannon_list = list(zip(*shannon_data))

    scores_df = pd.DataFrame({"mu": dist_list,
                              "sh_score": shannon_list})
    scores_df["selected"] = np.where(scores_df.sh_score == scores_df.sh_score[i_max], "sel", "non-sel")
    hue_order = ['sel', 'non-sel']
    sns.scatterplot(data=scores_df, x="mu", y="sh_score", hue="selected", palette=["green", "black"], alpha=0.8,
                    size="selected", sizes=(200, 500), ax=ax2, hue_order=hue_order)
    ax2.set_xlabel('distance (nm)', fontsize=45)
    ax2.set_ylabel('Scores', fontsize=45)
    sns.move_legend(
        ax1, "lower center",
        bbox_to_anchor=(.5, 1), ncol=2, title="{}\nFinal Distribution".format(dataset_name), frameon=False,
    )
    sns.move_legend(
        ax2, "lower center",
        bbox_to_anchor=(.5, 1), ncol=2, title="Bootstrap Scores", frameon=False,
    )

    plt.tight_layout()
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.savefig(figures_dir + f"{dataset_name}.pdf")
    plt.savefig(figures_dir + f"{dataset_name}.png")
    # plt.show()
    plt.clf()


def run_outlier_rejection(data, distances_list, dataset_name, results_dir, figures_dir, images_dir,
                          reject_lower, cutoff, bin_size, mu_ini=None, sigma_ini=None, dirty=False):
    """
    Main function to run outlier rejection
    Parameters
    ----------
    :param data: dataframe with data
    :param distances_list
    :param dataset_name
    :param results_dir
    :param figures_dir
    :param images_dir
    :param  mu_ini
    :param sigma_ini
    :param reject_lower: threshold to reject high scored distance distributions below this value
    (if you suspect your MLE results are fishy).
    :param cutoff: seek outliers in the 1 - cutoff tail of your initial distribution
    :param dirty: generate HTML files for each image with selected spots
    :param bin_size: size of bins for plots

    """
    # INPUT PARAMETERS
    np.random.seed(5)  # important when creating random datasets to ensure the results is always the same
    # By default, mu_ini and sigma_ini are None. The software starts with the initial guess of
    # mu0 = median of the distribution ; sigma0: standard deviation of the distribution. If you want to choose
    # your initial guess, change it in the options.py file
    if mu_ini is not None:
        mu0 = mu_ini
    else:
        mu0 = np.median(distances_list)  # initial guess for mu opt parameter
    if sigma_ini is not None:
        sigma0 = sigma_ini
    else:
        sigma0 = np.std(distances_list)  # initial guess for sigma opt parameter

    x0 = [mu0, sigma0]  # mu / sigma --> initial guess for "mu" and "sigma"
    print("\nChoosing distance distribution median and stdev as initial values to start fitting..\n\n"
          "\tInitial mu: {}\n"
          "\tInitial sigma: {}\n\n"
          "\tStarting optimization...\n".format(mu0, sigma0))

    # Create function to fit, based on Churchman., et al, 2006 (eq. 4)
    cumulative_density = cdf(mu0, sigma0)

    # compute the MLE on the data
    mu, sigma, sh_scores, i_min, i_max, n, sel_distribution = bootstrap(results_dir, LL,
                                                                        x0=x0, d=distances_list,
                                                                        reject_lower=reject_lower, cutoff=cutoff)
    data.loc[:, "selected"] = np.where(data.distances.isin(sel_distribution), "sel", "non-sel")

    # out.txt for final results, not beads
    with open(results_dir + "output.csv", "w") as out:
        out.write("mu,mu_err,sigma,sigma_err,n\n"
                  "{},{},{},{},{}\n".format(np.round(mu[i_max][0], 2),
                                            np.round(mu[i_max][1], 2),
                                            np.round(sigma[i_max][0], 2),
                                            np.round(sigma[i_max][1], 2),
                                            n))

    # Save figure with selected and non-selected spots based on KDE
    if dirty:
        for img_ in glob.glob(images_dir + "image_*"):
            image_number = img_.split("/")[-1].split(".")[0].split("_")[1]
            W1 = imread(img_)[0, :, :]
            W2 = imread(img_)[1, :, :]
            df_W1_final_sub = data[(data["img"] == image_number)]
            df_W2_final_sub = data[(data["img"] == image_number)]
            if df_W1_final_sub.shape[0] != 0:
                save_html_kde(figures_dir, W1, df_W1_final_sub, image_number, "W1")
            if df_W2_final_sub.shape[0] != 0:
                save_html_kde(figures_dir, W2, df_W2_final_sub, image_number, "W2")

    # Plot results
    plot_outlier_rejection(sel_distribution, cumulative_density, mu, sigma, i_max, n, sh_scores, dataset_name,
                           results_dir, figures_dir, bin_size)


def outlier_rejection(results_dir, figures_dir, images_dir, mu_ini, sigma_ini, reject_lower, cutoff,
                      bin_size, dirty=False):
    """
    Method to run outlier rejection of the distances'
    distribution, based on maximizing the likelihood
    estimate for "mu" and "sigma" to follow a distribution
    as described in Eq. 4 (Churchman, 2006).

    Parameters
    ----------
    :param results_dir
    :param figures_dir
    :param images_dir
    :param mu_ini: initial guess of mu value for starting the MLE
    :param sigma_ini: initial guess for sigma value for starting the MLE
    :param reject_lower: threshold to reject high scored distance distributions below this value
    (if you suspect your MLE results are fishy).
    :param cutoff: seek outliers in the 1 - cutoff tail of your initial distribution
    :param dirty: generate HTML files for each image with selected spots
    :param bin_size: size of bins for plots

    """
    print("\n\n####################################\n"
          "Initializing Outlier rejection \n"
          "########################################\n\n")
    # Read data from KDE to select distances
    # Check if detected spots are present
    if not os.path.exists(results_dir + 'gaussian_fit/') or len(os.listdir(results_dir + 'gaussian_fit/')) == 0:
        sys.stderr.write(
            '\nPICT-MODELLER-ERROR: Hey! Good morning! Are you trying to build the house from the roof?\n'
            'You should follow all the pipeline (segment, gaussian, kde) before entering here.\n'
            'Please, first process your data. \n'
            'Thanks! ;)\n\n')
        sys.exit(1)
    data = pd.concat(map(custom_outlier_read_csv, sorted(glob.glob(f"{results_dir}gaussian_fit/detected_gauss_*W1*"))),
                     ignore_index=True)
    distances = data.copy().distances.to_numpy()
    # print(len(distances))
    dataset_name = figures_dir.split("/")[-4]
    # RUN OUTLIER REJECTION
    run_outlier_rejection(data, distances, dataset_name, results_dir, figures_dir, images_dir,
                          reject_lower=reject_lower, mu_ini=mu_ini, sigma_ini=sigma_ini, cutoff=cutoff, dirty=dirty,
                          bin_size=bin_size)


if __name__ == "__main__":
    print("Functions for outlier rejection :)\n")
    sys.exit(0)
