#!/usr/bin/python3.7
# coding=utf-8
"""
Python functions to perform Gaussian fitting
to GFP & RFP spots of radius = 5 px
"""
import os
import sys
import time
import logging
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from skimage.io import imread
from lmfit import Parameters, minimize


# ======================
# GAUSSIAN FUNCTIONS
# ======================


def custom_gaussian_read_csv(file):
    """
    Function to read multiple csv (input data)
    """
    channel = file.split("/")[-1].split(".")[0].split("_")[3]
    img = file.split("/")[-1].split(".")[0].split("_")[2]
    df = pd.read_csv(file, sep="\t")  # sep="\t"
    if len(df) != 0:
        df.loc[:, "channel"] = channel
        df.loc[:, "img"] = img
        return df


def calculate_distances(df_1, df_2, px_size):
    """
    Calculate distances (in nm) between coloc. spots
    """
    return np.sqrt(
        (df_1.x.to_numpy() - df_2.x.to_numpy()) ** 2 + (df_1.y.to_numpy() - df_2.y.to_numpy()) ** 2) * px_size


def gaussian2D(x, y, cen_x, cen_y, sig_x, sig_y, offset):
    """
    Defining gaussian function to 2D data
    :param x:
    :param y:
    :param cen_x:
    :param cen_y:
    :param sig:
    :param offset:
    """
    return np.exp(-(((cen_x - x) / sig_x) ** 2 + ((cen_y - y) / sig_y) ** 2) / 2.0) + offset


def residuals(p, x, y, z):
    height = p["height"].value
    cen_x = p["centroid_x"].value
    cen_y = p["centroid_y"].value
    sigma_x = p["sigma_x"].value
    sigma_y = p["sigma_y"].value
    offset = p["offset"].value
    return z - height * gaussian2D(x, y, cen_x, cen_y, sigma_x, sigma_y, offset)


def clean_spot_boundaries(df, sub_df, image, radius):
    """
    Method to remove those spots close to the boundary of the image,
    from where it cannot be fitted to a gaussian distribution
    Parameters
    ----------
    df: raw dataframe
    sub_df: sub-dataframe of x,y coordinates
    image: image W1 or W2 as ndarray
    radius: mask radius to explore boundaries

    Returns
    -------
    cleaned dataframe
    """
    coords_to_remove = set()
    for coord in zip(sub_df.x.tolist(), sub_df.y.tolist()):
        y, x = int(coord[0]), int(coord[1])
        y_lower, y_upper = y - radius, y + radius
        x_lower, x_upper = x - radius, x + radius
        if y_lower < 0 or x_lower < 0 or y_upper > image.shape[0] or x_upper > image.shape[1]:
            print("Dropping coord {}".format(coord))
            df = df.drop(df[df.x == coord[0]].index)
            sub_df = sub_df.drop(sub_df[sub_df.x == coord[0]].index)
            coords_to_remove.add(coord[0])
    return df, sub_df


def slice_spot(image, coord, r=5, margin=0):
    """
    Slice spot in image by cropping with a radius and
    a margin.
    :param image: ndarray
    :param r: radius of the spot
    :param coord: array with x,y coordinates
    :param margin: margin to enlarge the spot by this 'margin'
    """
    try:
        coord = np.array(coord).astype(int)
        # y, x = coord[0], coord[1]
        # y_lower, y_upper = y - r, y + r
        # x_lower, x_upper = x - r, x + r
        # if y_lower > 0 or x_lower > 0 or y_upper < image.shape[0] or x_upper < image.shape[1]:
        mask = np.ones((r * 2 + margin, r * 2 + margin), dtype=np.int8)
        rect = tuple([slice(c - r, c + r) for c, r in zip(coord, (r, r))])
        slide = np.array(image[rect])
        return mask * slide
    except ValueError as ve:
        sys.stderr.write("{}".format(ve))


def gaussian_fit(spot):
    """
    Perform a gaussian fitting to normalized intensities in
    spot and evaluate the goodness of the fit with
    the R^2 value
    """
    # Create grid for x,y coordinates in spot
    xmin, xmax = 0, len(spot)
    ymin, ymax = 0, len(spot)
    x, y = np.meshgrid(np.arange(xmin, xmax), np.arange(ymin, ymax))

    # Set parameter for initial gaussian function
    initial = Parameters()
    initial.add("height", value=1., vary=False)  # height of the gaussian
    initial.add("centroid_x", value=5.)  # centroid x
    initial.add("centroid_y", value=5.)  # centroid y
    initial.add("sigma_x", value=2.5)  # centroid sigma
    initial.add("sigma_y", value=2.5)  # centroid sigma
    initial.add("offset", value=0., vary=False)  # centroid offset of the gaussian

    # Fit the intensity values to a gaussian function
    # residuals is the function to minimize with the initial function and the arg parameters
    fit = minimize(residuals, initial, args=(x, y, spot))
    # Calculate and return r-squared values to estimate goodness of the fit
    # Also return the sigma_x / sigma_y values, which gives us info about how round a spot is
    # Round spots should have a sigma ratio ~ 1.
    return 1 - fit.residual.var() / np.var(spot), fit.params["sigma_x"].value / fit.params["sigma_y"].value


def save_html_gaussian(path_to_save, channel_image, sub_df, img_num, channel_name):
    """
    Method to save image with scattered spots as html
    Parameters
    ----------
    channel_image: ndimage corresponding to channel W1 or W2
    path_to_save: path to save figure in html
    sub_df: sub-dataframe to work with
    img_num: image number
    channel_name: channel name ("W1" or "W2")

    """
    if not os.path.exists(path_to_save + "gaussian_fit/"):
        os.mkdir(path_to_save + "gaussian_fit/")

    selected = sub_df[sub_df["selected"] == "sel"]
    non_selected = sub_df[sub_df["selected"] == "non-sel"]
    percent_sel = round(len(selected) * 100 / (len(selected) + len(non_selected)), 3)
    foo_note = "<br>Number of Selected spots: {} / {} (Percentage = {} %)<br><br>".format(len(selected),
                                                                                          len(selected) + len(
                                                                                              non_selected),
                                                                                          percent_sel)
    # Create figure with lines to closest contour and closest neighbour
    fig_label_cont = px.imshow(channel_image, color_continuous_scale='gray',
                               title="<b>ImageMD {} {} - Gaussian selected</b><br>{}".format(img_num,
                                                                                             channel_name, foo_note))
    fig_label_cont.update_layout(coloraxis_showscale=False)  # to hide color bar

    # Plot spots with custom hover information
    fig_label_cont.add_scatter(x=selected["y"], y=selected["x"],
                               mode="markers",
                               marker=dict(color="green", size=7),
                               name="W1",
                               customdata=np.stack(([selected["r2_gaussian"],
                                                     selected["sigma_ratio"]]), axis=1),
                               hovertemplate=
                               '<b>x: %{x: }</b><br>'
                               '<b>y: %{y: } <b><br>'
                               '<b>r2_gauss: %{customdata[0]: }<b><br>'
                               '<b>sigma_ratio: %{customdata[1]: }<b><br>')

    fig_label_cont.add_scatter(x=non_selected["y"], y=non_selected["x"],
                               mode="markers",
                               marker=dict(color="red", size=7),
                               name="W2",
                               customdata=np.stack(([non_selected["r2_gaussian"],
                                                     non_selected["sigma_ratio"]]), axis=1),
                               hovertemplate=
                               '<b>x: %{x: }</b><br>'
                               '<b>y: %{y: } <b><br>'
                               '<b>r2_gauss: %{customdata[0]: }<b><br>'
                               '<b>sigma_ratio: %{customdata[1]: }<b><br>')
    # Save figure in output directory
    fig_label_cont.write_html(path_to_save + "gaussian_fit/" + "image_{}_{}.html".format(img_num, channel_name))


def main_gaussian(results_dir, images_dir, figures_dir, gaussian_cutoff, px_size, dirty=False):
    """
    2) Main method to run gaussian fitting on spots sorted from
    yeast segmentation method. Selection is based on R^2 (goddness
    of the gaussian fit).

    Parameters
    ------------
    :param images_dir: path to images directory for images BGN subtracted
    :param results_dir: path to results directory
    :param figures_dir: path to figures directory
    :param gaussian_cutoff: cutoff of the goodness of the gaussian fit
    :param px_size: pixel size of the camera
    :param dirty: generate HTML files for each image with selected spots
    """
    print("#############################\n"
          " Gaussian Fitting Selection \n"
          "#############################\n")
    logging.info("\n\n####################################\n"
                 "Initializing Gaussian Fitting Selection \n"
                 "########################################\n\n")
    # R^2 cutoff to reject spots under this cutoff (reject surely bad spots)
    r2_cutoff = gaussian_cutoff  # quality of spots above this r2 value

    # Load data from Segmentation Filter
    if not os.path.exists(f"{results_dir}kde/"):
        sys.stderr.write(f"PICT-MODELLER-ERROR: {results_dir}kde/ does not exists or is empty."
                         " Do you have spots detected after KDE? :/\n"
                         "Double check it and re-run!\n\n\tThanks and good luck! :)\n")
        sys.exit(1)
    data_seg_W1 = pd.read_csv(f"{results_dir}kde/W1_kde_sel.csv", sep='\t')
    data_seg_W2 = pd.read_csv(f"{results_dir}kde/W2_kde_sel.csv", sep='\t')
    percent_sel_total_W1 = list()
    percent_sel_total_W2 = list()
    percent_sel_total = list()
    total_data = 0
    total_selected = 0
    num_selected = 0
    for img_ in glob.glob(images_dir + "image_*.tif"):
        start = time.time()  # Keep track of time
        image_id = img_.split("/")[-1].split(".")[0].split("_")[1]
        # print(image_id)
        image_number = img_.split("/")[-1].split(".")[0].split("_")[1]
        print("Processing image {} ...\n".format(image_number))
        # Read image and separate frames
        image = imread(images_dir + "image_{}.tif".format(image_id))
        W1 = image[0]
        W2 = image[1]
        # Load spot coordinates for W1 and W2 for the given image
        # print(data_seg_W1.head())
        spots_df_W1 = data_seg_W1[data_seg_W1.img == image_id]
        spots_df_W2 = data_seg_W2[data_seg_W2.img == image_id]
        # Make sure that the number of spots in the image is not 0
        if len(spots_df_W1) != 0 and len(spots_df_W1) != 1:
            total_data += spots_df_W1.shape[0]

            # Slice dataframe with columns of interest (this is just for debugging)
            sub_df_W1 = spots_df_W1.loc[:, ["x", "y", "ID"]]
            sub_df_W2 = spots_df_W2.loc[:, ["x", "y", "ID"]]

            # Clean Spot Boundaries
            spots_df_W1, sub_df_W1 = clean_spot_boundaries(spots_df_W1, sub_df_W1, W1, radius=5)
            spots_df_W2, sub_df_W2 = clean_spot_boundaries(spots_df_W2, sub_df_W2, W2, radius=5)

            # Get coordinates for image and slice spots
            coords_W1 = list(zip(sub_df_W1.x.tolist(), sub_df_W1.y.tolist()))
            coords_W2 = list(zip(sub_df_W2.x.tolist(), sub_df_W2.y.tolist()))
            spots_W1 = [slice_spot(W1, coord) for coord in coords_W1]
            spots_W2 = [slice_spot(W2, coord) for coord in coords_W2]

            # Fit spot's normalized intensities to a gaussian distribution (see gaussian_fit function documentation)
            # Gaussian fit return: R^2 and sigma_ratio (sigma_r ~ 1 if the spot is round)
            r2_W1, sigma_r_W1 = list(
                zip(*[gaussian_fit((spot - spot.min()) / (spot.max() - spot.min())) for spot in spots_W1]))
            r2_W2, sigma_r_W2 = list(
                zip(*[gaussian_fit((spot - spot.min()) / (spot.max() - spot.min())) for spot in spots_W2]))

            # Add R^2 and sigma ratio to dataframes
            sub_df_W1.loc[:, "r2_gaussian"], sub_df_W1.loc[:, "sigma_ratio"] = r2_W1, sigma_r_W1
            sub_df_W2.loc[:, "r2_gaussian"], sub_df_W2.loc[:, "sigma_ratio"] = r2_W2, sigma_r_W2

            # Spot selection based on goodness of the fit, tested manually in different spots
            sub_df_W1.loc[:, 'selected'] = np.where(sub_df_W1["r2_gaussian"] > r2_cutoff, "sel", "non-sel")
            sub_df_W2.loc[:, 'selected'] = np.where(sub_df_W2["r2_gaussian"] > r2_cutoff, "sel", "non-sel")
            # Get percentage of selection for W1 and W2
            percent_sel_W1 = len(sub_df_W1[sub_df_W1["selected"] == "sel"]) * 100 / sub_df_W1.shape[0]
            percent_sel_W2 = len(sub_df_W2[sub_df_W2["selected"] == "sel"]) * 100 / sub_df_W2.shape[0]

            # Pair selected in W1 & W2
            selection_df_paired_W1 = spots_df_W1.loc[(sub_df_W1["selected"] == "sel") &
                                                     (sub_df_W2["selected"] == "sel")]
            selection_df_paired_W2 = spots_df_W2.loc[(sub_df_W2["selected"] == "sel") &
                                                     (sub_df_W1["selected"] == "sel")]
            # Assert shape W1 == shape W2
            assert selection_df_paired_W1.shape == selection_df_paired_W2.shape

            # write to log percentage of selection
            num_sel = selection_df_paired_W1.shape[0]
            # print(f"image {image_number}\n", num_sel, "\n", selection_df_paired_W1)

            num_selected += num_sel
            percent_sel = num_sel * 100 / spots_df_W1.shape[0]
            logging.info("\nImage {} --> {:02} / {:02} "
                         "spots selected.. --> {} %".format(image_number, num_selected, len(spots_df_W1), percent_sel))
            total_selected += num_selected

            if dirty:
                # Save figure with selected and non-selected spots based on goodness of the gaussian fit
                save_html_gaussian(figures_dir, W1, sub_df_W1, image_number, "W1")
                save_html_gaussian(figures_dir, W2, sub_df_W2, image_number, "W2")

            # Save df as csv: gaussian.csv
            if not os.path.exists(results_dir):
                os.mkdir(results_dir)
            if not os.path.exists(results_dir + "gaussian_fit/"):
                os.mkdir(results_dir + "gaussian_fit/")
            sub_df_W1.to_csv(results_dir + "gaussian_fit/" +
                             "all_gauss_{}_{}.csv".format(image_number, "W1"),
                             sep="\t", encoding="utf-8", header=True, index=False)
            sub_df_W2.to_csv(results_dir + "gaussian_fit/" +
                             "all_gauss_{}_{}.csv".format(image_number, "W2"),
                             sep="\t", encoding="utf-8", header=True, index=False)
            selection_df_paired_W1.to_csv(results_dir + "gaussian_fit/" +
                                          "detected_gauss_{}_W1.csv".format(image_number),
                                          sep="\t", encoding="utf-8", header=True, index=False)
            selection_df_paired_W2.to_csv(results_dir + "gaussian_fit/" +
                                          "detected_gauss_{}_W2.csv".format(image_number),
                                          sep="\t", encoding="utf-8", header=True, index=False)

            # Append percentages to list to write in report (log.txt)
            percent_sel_total_W1.append(percent_sel_W1)
            percent_sel_total_W2.append(percent_sel_W2)
            percent_sel_total.append(percent_sel)
            total_time = time.time() - start
            print("Image {} processed in {} s\n".format(image_number, round(total_time, 3)))
        else:
            print(f"\t--> 0 spots found in image {image_number}!\n")
            logging.info(f"\t--> 0 spots found in image {image_number}!\n")

    # WARNING!
    if len(percent_sel_total_W1) == 0 or len(percent_sel_total_W1) == 0 or len(percent_sel_total_W1) == 0:
        # WARNING: if no spots selected, we cannot continue!
        sys.stderr.write('PICT-MODELLER-ERROR: 0 spots found in datasets or very few :( '
                         'Probably due to a short dataset or poor quality images. \n'
                         '\t\tWe recommend a minimum number of input images == 20.\n'
                         '\t\tPlease, review your input image dataset and quality and run it again.\n\n'
                         '\tGood luck! :)\n\n')
        sys.exit(1)

    print("\n\nTotal Gauss-filtered W1 --> {} %\n"
          "Total Gauss-filtered W2 --> {} %\n\n"
          "Total Gauss-filtered --> {} % == {}\n".format(sum(percent_sel_total_W1) / len(percent_sel_total_W1),
                                                         sum(percent_sel_total_W2) / len(percent_sel_total_W2),
                                                         sum(percent_sel_total) / len(percent_sel_total),
                                                         num_selected))

    logging.info("\n\nTotal Gauss-filtered W1 --> {} %\n"
                 "Total Gauss-filtered W2 --> {} %\n\n"
                 "Total Gauss-filtered --> {} % \n".format(sum(percent_sel_total_W1) / len(percent_sel_total_W1),
                                                           sum(percent_sel_total_W2) / len(percent_sel_total_W2),
                                                           sum(percent_sel_total) / len(percent_sel_total)))
    #####################################
    # PLOT SELECTED SPOTS AFTER GAUSSIAN
    #####################################
    print("\nPlotting Gaussian selection....\n")
    # Load data ensuring that W1 & W2 are paired
    df_W1 = pd.concat(map(custom_gaussian_read_csv, sorted(glob.glob(results_dir + "gaussian_fit/all*W1*"))),
                      ignore_index=True)
    df_W2 = pd.concat(map(custom_gaussian_read_csv, sorted(glob.glob(results_dir + "gaussian_fit/all*W2*"))),
                      ignore_index=True)
    # Combine W1&W2 data into a df and label selected
    df_data = pd.concat([df_W1.r2_gaussian.rename("r2_W1"), df_W2.r2_gaussian.rename("r2_W2")], axis=1)
    df_data.loc[:, 'selected'] = np.where((df_W1["r2_gaussian"] >= r2_cutoff) & (df_W2["r2_gaussian"] >= r2_cutoff),
                                          "sel", "non-sel")
    df_data.loc[:, 'channel'] = np.where((df_W1["channel"] == "W1") & (df_W2["channel"] == "W2"),
                                         "W1", "W2")
    # Plot values in the R^2 space
    fig, ax = plt.subplots(constrained_layout=True, figsize=(8, 8))
    hue_order = ['sel', 'non-sel']
    sns.scatterplot(data=df_data, x="r2_W1", y="r2_W2", hue="selected", palette=["red", "black"], alpha=0.6,
                    s=50, zorder=10, ax=ax, hue_order=hue_order)
    # ax.set_title("Goodness of the Gaussian Fit", fontweight="bold", size=20)
    ax.set_ylabel("$R^{2} _{GFP}$", fontsize=20)
    ax.set_xlabel("$R^{2} _{RFP}$", fontsize=20)
    ax.set(xlim=(0, 1))
    ax.set(ylim=(0, 1))

    fig.savefig(figures_dir + "gaussian.png", dpi=150)
    plt.clf()

    # MEASURE DISTANCE DISTRIBUTION AFTER GAUSSIAN
    initial_distances = np.loadtxt(results_dir + "distances_after_warping.csv")
    df_W1 = pd.concat(map(custom_gaussian_read_csv, sorted(glob.glob(results_dir + "gaussian_fit/detected_gauss*W1*"))),
                      ignore_index=True)
    df_W2 = pd.concat(map(custom_gaussian_read_csv, sorted(glob.glob(results_dir + "gaussian_fit/detected_gauss*W2*"))),
                      ignore_index=True)
    distances_gauss = calculate_distances(df_W1, df_W2, px_size=px_size)
    np.savetxt(results_dir + "gaussian_fit/gauss_distances.csv", distances_gauss, delimiter=",")
    # PLOT NEW DISTANCE DISTRIBUTION
    fig, ax = plt.subplots(figsize=(25, 15))
    sns.set(font_scale=3)
    ax.set_title("Distances after GAUSSIAN selection\n\n"
                 "mean detection = {} nm; stdev detection = {} nm; n = {}\n"
                 "mean gaussian = {} nm; stdev gaussian = {} nm; "
                 "n = {} \n".format(np.around(np.mean(initial_distances), 2),
                                    np.around(np.std(initial_distances), 2),
                                    len(initial_distances),
                                    np.around(np.mean(distances_gauss), 2),
                                    np.around(np.std(distances_gauss), 2),
                                    len(distances_gauss)),
                 fontweight="bold", size=25)
    sns.histplot(data=initial_distances, kde=True, color="tab:grey", ax=ax, fill=True, stat="density")
    sns.histplot(data=distances_gauss, kde=True, ax=ax, color="tab:red", fill=True, stat="density")
    ax.set_xlabel("Distances (nm) ", fontsize=45, labelpad=30)
    ax.set_ylabel("Density ", fontsize=45, labelpad=30)
    ax.tick_params(axis='x', labelsize=30)
    ax.tick_params(axis='y', labelsize=30)
    # ax.axvline(x=np.mean(initial_distances), color='sandybrown', ls='--', lw=2.5, alpha=0.8)
    # ax.axvline(x=np.mean(distances_gauss), color='cornflowerblue', ls='--', lw=2.5, alpha=0.8)
    plt.savefig(figures_dir + "gaussian_fit/" + "distances_after_gauss.png")
    plt.clf()

    return total_data, total_selected


if __name__ == "__main__":
    print("Gaussian Fit functions :)\n")
    sys.exit(0)
