import os
import shutil
import sys
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import trackpy as tp
import pims
import numpy as np
from skimage import io, util
from scipy import stats
from scipy import spatial
import plotly.express as px
import plotly.graph_objects as go
from pymicro.view.vol_utils import compute_affine_transform
import glob

###########
# ALGEBRA #
###########


def get_data_from_1d_grid(x, x_min, dx, indexes):
    """
    Get data from a 1d-vector with a list of indexes.

    x: vector with values in x axis.
    dx: pixel size of the grid.
    indexes: desired indexes to select the data from.
    """
    selected_data = list()
    for i in indexes:
        ix = i[0]
        x_values = x[(x > x_min + (ix * dx)) & (x < x_min + (ix + 1) * dx)]
        selected_data += list(x_values.tolist())
    return selected_data


def min_mass_selection(data, bins=1000, min_mass_cutoff=0.01):
    """
    Select those spots that fall in the area below the threshold percentage
    :param data: data in ndarray
    :param bins: bin size
    :param min_mass_cutoff: in tant per 1
    """
    # Discard spots that fall in the 5% of spots with less mass
    # with respect to the total mass in the image
    dx = (data.max() - data.min()) / bins  # jump to explore the left tail of the distribution
    ix = 1
    low_mass_area = np.sum(data[data <= (data.min() + (dx * ix))])
    low_mass_percent = low_mass_area / np.sum(data)
    cf = data.min() + dx
    if low_mass_percent < min_mass_cutoff:
        ix += 1
        while not low_mass_percent > min_mass_cutoff:
            low_mass_area = np.sum(data[data <= (data.min() + (dx * ix))])
            low_mass_percent = low_mass_area / np.sum(data)
            cf = data.min() + (dx * ix)
            ix += 1
    return np.sort(data[data <= cf])


def select_mass_cdf(data, bins=100, min_mass_cutoff=0.01, max_mass_cutoff=0.90, debug=False, verbose=False):
    """
    Estimates the probability density function (PDF) and the cumulative probability
    density function (CDF) of a univariate datapoints. In this case, the function helps
    to reject the m % of spots that are too bright (default: 0.85).

    data: 1d-array of values which are the mass of detected spots.
    bins: smooth parameter to calculate the density (sum of density ~ 1)
    max_mass_cutoff: Selecting spots below this threshold (in tant per 1)
    min_mass_cutoff: Selecting spots above this threshold (in tant per 1)
    """
    # Low-mass spots selection
    low_mass_spots = min_mass_selection(data, bins=1000, min_mass_cutoff=min_mass_cutoff)
    # Use a gaussian kernel to estimate the density
    kernel = stats.gaussian_kde(data, bw_method="silverman")
    positions = np.linspace(data.min(), data.max(), bins)
    dx = positions[1] - positions[0]
    pdf = kernel(positions) * dx
    total_pdf = np.sum(pdf)
    if 0.98 < total_pdf <= 1:
        if verbose:
            print("Sum pdf = {}\n".format(bins, total_pdf))
        pass
    else:
        if debug:
            print("\n++ DEBUG: Optimizing bin size...\n"
                  "\t+ Original bin size = {}\n"
                  "\t+ Sum pdf = {}\n".format(bins, total_pdf))
        search_bin_step = - 5
        while not 0.98 < total_pdf <= 1:
            if bins <= 10:
                break
            bins += search_bin_step
            # print(bins)
            positions = np.linspace(data.min(), data.max(), bins)
            dx = positions[1] - positions[0]
            pdf = kernel(positions) * dx
            total_pdf = np.sum(pdf)
            if total_pdf > 1:
                search_bin_step = 2
        if debug:
            print("\n++ DEBUG: Bin size optimized!\n"
                  "\t+ New bin size = {}\n"
                  "\t+ Sum pdf = {}\n".format(bins, total_pdf))

    density_sorted = np.copy(pdf)
    density_sorted[::-1].sort()
    cum = np.cumsum(density_sorted)
    # Select indexes based on cumulative probability cutoff
    sel_idx = [np.where(pdf == index) for index in density_sorted[cum <= max_mass_cutoff]]
    selected = np.sort(np.asarray(get_data_from_1d_grid(data, np.min(positions), dx, sel_idx)))
    selected = selected[selected >= low_mass_spots.max()]
    return selected


def calculate_distances(df_1, df_2, px_size):
    """
    Calculate distances (in nm) between coloc. spots
    """
    return np.sqrt(
        (df_1.x.to_numpy() - df_2.x.to_numpy()) ** 2 + (df_1.y.to_numpy() - df_2.y.to_numpy()) ** 2) * px_size


# Distance distribution
def ddist(coords1, coords2, px_size):
    """
    Return list of distances
    """
    return np.sqrt(np.square(coords1[:, 0] - coords2[:, 0]) + np.square(coords1[:, 1] - coords2[:, 1])) * px_size


# Calculate Distance
def distance(coord1, coord2, px_size):
    """
    Return distance in nm
    """
    return np.round(np.sqrt(np.square(coord1[0] - coord2[0]) + np.square(coord1[1] - coord2[1])) * px_size, 3)


# reduce data for testing
def reduce_data(arr, lw, ub):
    """
    Crop image for testing
    """
    return arr[(arr[:, 0] > lw) & (arr[:, 0] < ub) & (arr[:, 1] > lw) & (arr[:, 1] < ub)]


def normal(mean, std, color="black"):
    x = np.linspace(mean - 4 * std, mean + 4 * std, 20)
    p = stats.norm.pdf(x, mean, std)
    z = plt.plot(x, p, color, linewidth=2)


###########
# PLOTTING #
###########


def plot_mass(path_to_save, df, image_name):
    """
    Plot intensity distribution for each frame
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    hue_order = ['sel', 'non-sel']
    sns.histplot(data=df, x="mass", hue="selected", palette=["red", "black"], kde=True, alpha=0.2,
                 stat="count", fill=True, ax=ax1, hue_order=hue_order)
    sns.kdeplot(x=df["mass"], y=df["ecc"], fill=True, thresh=0.05, cbar=False, ax=ax2,
                bw_method="silverman", hue_order=hue_order)
    sns.scatterplot(data=df, x="mass", y="ecc", hue="selected", palette=["red", "black"], alpha=0.2,
                    size="selected", sizes=(100, 50), ax=ax2, hue_order=hue_order)
    plt.tight_layout()
    # plt.show()
    if not os.path.exists(path_to_save):
        os.mkdir(path_to_save)
    plt.savefig(path_to_save + "mass_selection_{}.png".format(image_name))


def plot_distance_distribution(path_to_save, df, image_name):
    """
    Method to plot ditance distribution of bead
    pairs BEFORE and AFTER warping
    Parameters
    ----------
    path_to_save
    df
    image_name

    Returns
    -------

    """
    fig, ax = plt.subplots(figsize=(15, 15))
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.despine()
    ax.set_title("Beads Distances BEFORE Warping\n"
                 "Median = {}\n"
                 "Stdev = {}\n".format(df.distances.median(), df.distances.std()), fontweight="bold", size=25)
    sns.histplot(data=df, x="distances", kde=True, ax=ax, fill=True, stat="density")
    ax.axvline(x=df.distances.mean(), color='red', ls='--', lw=2.5, alpha=0.1)
    plt.tight_layout()
    plt.savefig(path_to_save + "/distances_{}.png".format(image_name))


def save_html_selected_detection(path_to_save, spots_df, image_name, ndimage, percentile, min_mass_cutoff,
                                 max_mass_percent):
    """
    Display selected and non-selected spots in an interactive image
    and save image in html format
    Parameters
    ----------
    image_name: channel name: "W1" or "W2"
    percentile: percentile for selecting bright spots
    max_mass_percent: to reject those m% of spots that are too bright (clusters).
    min_mass_cutoff: discard low mass spots below this threshold
    spots_df: dataframe with spots coordinates for a given image
    ndimage: image (ndarray)
    path_to_save: path to save image
    """
    # Check path and / or create it
    if not os.path.exists(path_to_save):
        os.mkdir(path_to_save)
    for f in range(ndimage.shape[0]):
        channel_name = "W1" if f == 0 else "W2"
        # Selected and non selected spots do display on images
        selected = spots_df[(spots_df["frame"] == f) & (spots_df["selected"] == "sel")]
        non_selected = spots_df[(spots_df["frame"] == f) & (spots_df["selected"] == "non-sel")]
        # Foo note information for the image
        foo_note = "<br>Number of Selected spots: {}<br>" \
                   "Number of Non-selected spots: {}<br>" \
                   "Percentile: {}%<br>" \
                   "Max mass cutoff: {}%<br>" \
                   "Low mass cutoff: {}%<br>".format(selected.shape[0], non_selected.shape[0],
                                                     percentile, round((1 - max_mass_percent), 2) * 100,
                                                     min_mass_cutoff * 100)

        fig_label_cont = px.imshow(ndimage[f, :, :], color_continuous_scale='gray',
                                   title="<b>Image {} <br> Frame {} - "
                                         "Spot Selection</b><br>{}".format(channel_name, image_name, foo_note))
        fig_label_cont.update_layout(coloraxis_showscale=False)  # to hide color bar

        # Plot Selected & Non - selected spots with custom hover information
        fig_label_cont.add_scatter(x=selected["x"], y=selected["y"],
                                   mode="markers",
                                   marker_symbol="circle-open-dot",
                                   marker=dict(color="green", size=15,
                                               line=dict(width=2,
                                                         color='floralwhite')),
                                   name="selected",
                                   opacity=0.4,
                                   customdata=np.stack(([selected["mass"],
                                                         selected["size"],
                                                         selected["ecc"],
                                                         selected["signal"]]), axis=1),
                                   hovertemplate=
                                   '<b>x: %{x: }</b><br>'
                                   '<b>y: %{y: } <b><br>'
                                   '<b>mass: %{customdata[0]: }<b><br>'
                                   '<b>size: %{customdata[1]: }<b><br>'
                                   '<b>ecc:  %{customdata[2]: }<b><br>'
                                   '<b>SNR:  %{customdata[3]: }<b><br>')

        fig_label_cont.add_scatter(x=non_selected["x"], y=non_selected["y"],
                                   mode="markers",
                                   marker_symbol="circle-open-dot",
                                   marker=dict(color="red", size=10,
                                               line=dict(width=2,
                                                         color='floralwhite')),
                                   name="selected",
                                   opacity=0.4,
                                   customdata=np.stack(([non_selected["mass"],
                                                         non_selected["size"],
                                                         non_selected["ecc"],
                                                         non_selected["signal"]]), axis=1),
                                   hovertemplate=
                                   '<b>x: %{x: }</b><br>'
                                   '<b>y: %{y: } <b><br>'
                                   '<b>mass: %{customdata[0]: }<b><br>'
                                   '<b>size: %{customdata[1]: }<b><br>'
                                   '<b>ecc:  %{customdata[2]: }<b><br>'
                                   '<b>SNR:  %{customdata[3]: }<b><br>')

        fig_label_cont.write_html(path_to_save + "/selected_{}_{}.html".format(image_name, channel_name))
        fig_label_cont.data = list()


def save_html_detection(path_to_save, spots_df, image_name, ndimage, percentile):
    """
    Display detected spots in an interactive image
    and save image in html format. Spots df does
    not have the "selected" column.
    Parameters
    ----------
    image_name: channel name: "W1" or "W2"
    percentile: percentile for selecting bright spots
    spots_df: dataframe with spots coordinates for a given image
    ndimage: image (ndarray)
    path_to_save: path to save image
    """
    # Check path and / or create it
    if not os.path.exists(path_to_save):
        os.mkdir(path_to_save)
    for f in range(ndimage.shape[0]):
        channel_name = "W1" if f == 0 else "W2"
        spots_df_channel = spots_df[spots_df["frame"] == f]

        # Foo note information for the image
        foo_note = "<br>Number of Detected spots: {}<br>" \
                   "Percentile: {}%<br>".format(spots_df_channel.shape[0], percentile)

        fig_label_cont = px.imshow(ndimage[f, :, :], color_continuous_scale='gray',
                                   title="<b>Image {} <br> Frame {} - "
                                         "Spot Selection</b><br>{}".format(channel_name, image_name, foo_note))
        fig_label_cont.update_layout(coloraxis_showscale=False)  # to hide color bar

        # Plot Selected & Non - selected spots with custom hover information
        fig_label_cont.add_scatter(x=spots_df_channel["y"], y=spots_df_channel["x"],
                                   mode="markers",
                                   marker_symbol="circle-open-dot",
                                   marker=dict(color="green", size=15,
                                               line=dict(width=2,
                                                         color='floralwhite')),
                                   name="selected",
                                   opacity=0.4,
                                   customdata=np.stack(([spots_df_channel["mass"],
                                                         spots_df_channel["size"],
                                                         spots_df_channel["ecc"],
                                                         spots_df_channel["signal"]]), axis=1),
                                   hovertemplate=
                                   '<b>x: %{x: }</b><br>'
                                   '<b>y: %{y: } <b><br>'
                                   '<b>mass: %{customdata[0]: }<b><br>'
                                   '<b>size: %{customdata[1]: }<b><br>'
                                   '<b>ecc:  %{customdata[2]: }<b><br>'
                                   '<b>SNR:  %{customdata[3]: }<b><br>')

        fig_label_cont.write_html(path_to_save + "/detected_{}_{}.html".format(image_name, channel_name))
        fig_label_cont.data = list()


# Plot coordinates with interactive plotly
def plotly_coords(c1_coords, c2_coords, path=None, c1_corrected=None, title=""):
    """
    Scatter plot of 2C coordinates
    """
    x_coords_c1, y_coords_c1 = c1_coords[:, 0], c1_coords[:, 1]
    x_coords_c2, y_coords_c2 = c2_coords[:, 0], c2_coords[:, 1]
    fig = px.scatter(title=title).add_scatter(x=x_coords_c1, y=y_coords_c1, mode="markers", name="W1",
                                  marker_symbol="square-dot",
                                  marker=dict(color="red", size=5,
                                                line=dict(width=2,
                                                          color='red')))
    fig.add_scatter(x=x_coords_c2, y=y_coords_c2, mode="markers", name="W2",
                    marker_symbol="circle-dot",
                    marker=dict(color="green", size=5,
                                line=dict(width=2,color='green')))
    if c1_corrected is not None:
        x_corr, y_corr = c1_corrected[:, 0], c1_corrected[:, 1]
        fig.add_scatter(x=x_corr, y=y_corr, mode="markers",
                name="w1_warped_test", marker_symbol="x-dot",
                marker=dict(color="orange", size=5,
                                            line=dict(width=2,
                                                      color='orange')))
    fig.update_layout(xaxis=dict(showgrid=False),
                  yaxis=dict(showgrid=False), template='plotly_dark')
    fig.update_yaxes(autorange="reversed")
    # Draw lines between W2 - W1 before warping to see link
    for i in range(len(x_coords_c1)):
        x1, x2 = x_coords_c1[i], x_coords_c2[i]
        y1, y2 = y_coords_c1[i], y_coords_c2[i]
        fig.add_trace(go.Scatter(x=[x1, x2], y=[y1, y2],
                                 mode="lines",
                                 line=go.scatter.Line(color="white"),
                                 name="{}".format(i), showlegend=False))
    if path is not None:
        fig.write_html(path + "/input/beads_{}.html".format(title))
        fig.data = list()
    else:
        fig.show()


###########
# PROCESS #
###########


def create_beads_stacks(path_to_beads):
    """
    Split beads W1 and W2 in individual frames and stack
    frame_x_W1 with frame_x_W2 and save in path directory
    """
    # Only stack frames if the number of stacked frames in directory is not 4
    # which means that, since we have 4 frames (tetrastack), we should have 4 pairs
    if len(glob.glob(path_to_beads + "/frame*.tif")) != 4:
        print('\tCreating stacks W1-W2...\n ')
        beads_W1 = io.imread(path_to_beads + "/W1.tif")
        beads_W2 = io.imread(path_to_beads + "/W2.tif")
        for f in range(beads_W2.shape[2]):
            frame_pair = np.stack([beads_W1[:, :, f], beads_W2[:, :, f]])
            io.imsave(path_to_beads + "/frame_{}.tif".format(f), frame_pair, plugin="tifffile", check_contrast=False)
    else:
        pass


def detection(path_to_save, image_name, ndimage, pims_frames, percentile, min_mass_cutoff, max_mass_cutoff,
              verbose=True, test=False):
    """
    Spot selection using Trackpy.

    image_name: name to save image in directory
    ndimage: image frame in a ndarray format
    pims_frames: PIMS frame/s
    percentile:
    min_mass_cutoff:
    max_mas_cutoff:
    """
    # SPOT DETECTION
    f = tp.batch(pims_frames[:], diameter=11, percentile=percentile, engine='numba')
    f.loc[:, "ID"] = list(range(1, f.shape[0] + 1))
    f.loc[:, 'size'] = f['size'].apply(lambda x: x ** 2)  # remove sqrt from size formula
    # Select spots with a cumulative density probability less than a threshold
    test_mass_selection = select_mass_cdf(f.mass.to_numpy(), bins=100, min_mass_cutoff=min_mass_cutoff,
                                          max_mass_cutoff=max_mass_cutoff, verbose=verbose, debug=False)
    f['selected'] = np.where(f.mass.isin(test_mass_selection), "sel", "non-sel")
    if verbose:
        print("\nInitial number of spots detected by trackpy: {}\n"
              "Number of spots discarded regarding mass: {}\n"
              "Final number of selected spots: {}\n\n".format(f.shape[0],
                                                              len(f[
                                                                      f["selected"] ==
                                                                      "non-sel"]),
                                                              len(test_mass_selection)))
    # SAVING RESULTS
    if not os.path.exists(path_to_save):
        os.mkdir(path_to_save)
    if test:
        if not os.path.exists(path_to_save + "selection_test/detected_spots/"):
            os.mkdir(path_to_save + "selection_test/detected_spots/")
        # PLOT: (mass according to selection) and html with selected spots for each channel
        plot_mass(path_to_save + "selection_test/detected_spots/", f, image_name)
        save_html_selected_detection(path_to_save + "selection_test/detected_spots/", f, image_name, ndimage, percentile,
                                     max_mass_cutoff, min_mass_cutoff)
    else:
        # Plot selected spots
        plot_mass(path_to_save + "/intensities/", f, image_name)
        save_html_selected_detection(path_to_save + "/detected_spots/", f, image_name, ndimage, percentile,
                                     min_mass_cutoff, max_mass_cutoff)
    return f


def linking(path_to_beads, f_batch_selection, image_name, ndarray, link, percentile, min_mass_cutoff, max_mass_cutoff,
            test=False):
    """
    Linking (pairing) detected particles from two channels (0/red/W1 and 1/green/W2)
    Selecting only paired detections between two channels and saving selections to files.
    """
    t = tp.link(f_batch_selection, search_range=link, pos_columns=["x", "y"])
    t = t.sort_values(by=["particle", "frame"])  # sort particle according to "particle" and "frame"
    t = t[t.duplicated("particle", keep=False)]  # select paired only
    f_batch = f_batch_selection.copy()
    f_batch.loc[:, "selected"] = np.where(f_batch.ID.isin(t.ID.tolist()), "sel",
                                          "non-sel")
    # PLOT selected after linking the two channels
    save_html_selected_detection(path_to_beads + "/linked_spots/", f_batch, image_name, ndarray,
                                 percentile, min_mass_cutoff, max_mass_cutoff)

    # Separate frame0 and frame1 in two df
    t_W1 = t[t["frame"] == 0]
    t_W2 = t[t["frame"] == 1]

    # Save coordinates in separate csv files
    t_W1[["x", "y"]].to_csv(path_to_beads + "/linked_spots/" +
                            "detected_{}_W1.csv".format(image_name.split("_")[1]),
                            sep="\t",
                            encoding="utf-8", header=True, index=False)
    t_W2[["x", "y"]].to_csv(path_to_beads + "/linked_spots/" +
                            "detected_{}_W2.csv".format(image_name.split("_")[1]),
                            sep="\t",
                            encoding="utf-8", header=True, index=False)
    # PLOT INITIAL DISTANCE DISTRIBUTION
    t_W1, t_W2 = t_W1.copy(), t_W2.copy()
    t_W1.loc[:, "distances"] = calculate_distances(t_W1, t_W2)
    t_W2.loc[:, "distances"] = calculate_distances(t_W1, t_W2)
    if test:
        if not os.path.exists(path_to_beads + "selection_test/linked_spots/"):
            os.mkdir(path_to_beads + "selection_test/linked_spots/")
        plot_distance_distribution(path_to_beads + "selection_test/linked_spots/", t_W1, image_name)
    else:
        if not os.path.exists(path_to_beads + "/linked_spots/"):
            os.mkdir(path_to_beads + "/linked_spots/")
        plot_distance_distribution(path_to_beads + "/linked_spots/", t_W1, image_name)

    return t_W1, t_W2


def prepare_data(working_dir, path_to_beads, rfp_channel, gfp_channel):
    """
    Method for checking the data before running the protocol:
    - input/
        beads/
            frame_0.tif ...
    - output/
    Parameters
    ----------
    working_dir: path of working directory
    path_to_beads: path to beads
    rfp_channel: int (default: 0)
    gfp_channel: int (default: 1)
    """
    print("\nChecking your data structure...\n")
    if not os.path.exists(working_dir + "input"):
        os.mkdir(working_dir + "input")
        print("\t- input/ folder created!\n")
    elif not os.path.exists(path_to_beads):
        os.mkdir(path_to_beads)
        print("\t- input/beads/ folder created!\n")
    else:
        print("\t- input/ and input/beads already present.")
    # raw beads are there? Then, move to input/beads and rename
    if len(glob.glob(f"{working_dir}/*Pos*.ome.tif")) != 0:
        if len(glob.glob(f"{working_dir}/*Pos*.ome.tif")) == 4:
            for bead_img in glob.glob(f"{working_dir}/*Pos*.ome.tif"):
                new_name = bead_img.split("/")[-1].split("_")[-1].split(".")[0].replace("Pos", "frame_") + ".tif"
                print(new_name)
                os.rename(bead_img, f"{working_dir}/{new_name}")
                shutil.move(f"{working_dir}/{new_name}", f"{path_to_beads}/{new_name}")
    # Create stack W1 and W2 if not present
    if not os.path.exists(path_to_beads + "W1.tif") and not os.path.exists(path_to_beads + "W2.tif"):
        print('\n\t- Creating stacks W1.tif and W2.tif ...\n ')
        W1_frames = list()
        W2_frames = list()
        for bead_file in sorted(glob.glob(path_to_beads + "frame*.tif")):
            pos = int(bead_file.split("/")[-1].split(".")[0].split("_")[1])
            b_img = io.imread(bead_file)
            stack_index = [b_img.shape.index(f) for f in b_img.shape if f < 5][0]
            if stack_index == 0:
                W1_frames.append(b_img[rfp_channel, :, :])
                W2_frames.append(b_img[gfp_channel, :, :])
                # io.imsave(path_to_beads + f"red_{pos}.tif", b_img[rfp_channel, :, :], plugin="tifffile",
                #           check_contrast=False)
                # io.imsave(path_to_beads + f"green_{pos}.tif", b_img[rfp_channel, :, :], plugin="tifffile",
                #           check_contrast=False)
            if stack_index == 2:
                W1_frames[pos, :, :] += b_img[:, :, rfp_channel]
                W2_frames.append(b_img[:, :, gfp_channel])
        W1 = np.stack(W1_frames, axis=2)
        W2 = np.stack(W2_frames, axis=0)
        io.imsave(path_to_beads + "W1.tif", W1, plugin="tifffile", check_contrast=False)
        io.imsave(path_to_beads + "W2.tif", W2, plugin="tifffile", check_contrast=False)
    print("\n\t...DONE!\n")


def piecewise_affine(query_spot, c1_ref_beads, c2_ref_beads, search_range, min_candidates=10):
    """
    Apply piecewise affine transform to query spot
    using 2C reference.
    Returns the local refined coordinates of the query.
    """
    search = True
    # Search closest neighbors with a minimum number of candidates
    # Keep the search until it gets the proper distance to get the min number of candidates
    while search:
        # Get closest neighbors
        d, ids = spatial.KDTree(c2_ref_beads).query(query_spot, search_range)
        ids = ids[d < search_range]
        if len(ids) >= min_candidates:
            search = False
        else:
            search_range += 10

    candidates_c1 = c1_ref_beads[ids]
    candidates_c2 = c2_ref_beads[ids]
    # compute the AFFINE transform & translation from the REF point set
    translation, transformation = compute_affine_transform(candidates_c2, candidates_c1)

    # TRANSFORMATION
    ref_centroid = np.mean(candidates_c2, axis=0)
    mov_centroid = np.mean(candidates_c1, axis=0)
    refined_spot = ref_centroid + np.dot(transformation, query_spot - mov_centroid)
    # print(f"Query spot {query_spot} with search {search_range} and n {len(ids)}\n")
    return refined_spot


def get_coords(path_to_save, path_to_beads, beads_head, separation, percentile,
               min_mass_cutoff, max_mass_cutoff, verbose=True):
    """
    Method to get coordinates from a set of 2C (2 channel)
    images of beads defined in the path

    Args:
        path_to_input: path to input folder
            e.g. tree:
            input/
                beads_registration/
                beads_test/
        path_to_beads: path to bead images (ref or test)
        beads_head: pattern in the names of beads images (e.g. beads_*)
        percentile: selecting spots in the upper bound of this percentile
        min_mass_cutoff
        max_mass_cutoff
        beads_only: running only on the bead distances (short - close to zero),
         when option is -o beads.
    """
    if verbose:
        print("\n#############################\n"
              "     BEADS REGISTRATION \n"
              "#############################\n")
    x_coords_W1 = list()
    y_coords_W1 = list()
    x_coords_W2 = list()
    y_coords_W2 = list()

    # Everything is ok now for the analysis
    for img in glob.glob(path_to_beads + beads_head):
        # READ IMAGE, set name and image number, use PIMS to read frames
        name = img.split("/")[-1].split(".")[0]
        frames = pims.open(img)

        # SPOT DETECTION
        f_batch = detection(path_to_save, name, io.imread(img), frames, percentile, min_mass_cutoff, max_mass_cutoff,
                            verbose)

        # LINK SELECTION
        f_batch_sel = f_batch[f_batch['selected'] == "sel"].drop(columns=["selected"])
        paired_df_W1, paired_df_W2 = linking(path_to_save, f_batch_sel, name, io.imread(img), separation, percentile,
                                             min_mass_cutoff, max_mass_cutoff)

        # Append x and y coordinates to lists
        x_coords_W1 += paired_df_W1.x.tolist()
        y_coords_W1 += paired_df_W1.y.tolist()
        x_coords_W2 += paired_df_W2.x.tolist()
        y_coords_W2 += paired_df_W2.y.tolist()

    # Ref and Mov coordinates
    c2_coords = np.asarray(list(zip(x_coords_W2, y_coords_W2)))
    c1_coords = np.asarray(list(zip(x_coords_W1, y_coords_W1)))

    # Save coords to csv
    np.savetxt(path_to_save + "/coords_W1.csv", c1_coords, delimiter=",", header="x,y")
    np.savetxt(path_to_save + "/coords_W2.csv", c2_coords, delimiter=",", header="x,y")

    return c2_coords, c1_coords


if __name__ == "__main__":





    # beads_path = "../Exo70-Sec5/input/beads"
    # test_path = "../Exo70-Sec5/input/test"
    # pc = 80  # in tant per cent
    # max_mass = 0.95  # in tant per 1
    # min_mass = 0.01  # in tant per 1
    # # Separate frames and create pair stack of beads
    # # create_beads_stacks(f"{beads_path}")
    #
    # verbose = True
    #
    # # Define REF and TEST
    # # REF (bead image)--> ch1/ch2 (mov/ref) channels to create transformation matrices
    # # TEST  (bead image)--> ch1/ch2 to test the registration error
    # c2_ref, c1_ref = get_coords(beads_path, f"{beads_path}/frame*.tif", pc, min_mass, max_mass)
    # c2_test, c1_test = get_coords(test_path, f"{test_path}/*.tif",  pc, min_mass, max_mass)
    # # Plot REF & TEST coords
    # plotly_coords("../Exo70-Sec5", c1_ref, c2_ref, title=f"2C REF coordinates\n N={len(c2_ref)}")
    # plotly_coords("../Exo70-Sec5", c1_test, c2_test, title=f"2C TEST coordinates\n N={len(c2_test)}")
    # # Get new coords
    # c1_test_new = np.empty_like(c1_test)
    # for i in range(len(c1_test)):
    #     c1_test_new[i] = piecewise_affine(c1_test[i], c1_ref, c2_ref, search_range=40, min_candidates=5)
    # # Plotly
    # plotly_coords("../Exo70-Sec5", c1_test, c2_test, c1_test_new, title=f"2C TEST WARPED\n N={len(c1_test)}")
    #
    # # Plot Distance distribution
    # fig, ax = plt.subplots()
    # sns.set(font_scale=1)
    # sns.set_style("whitegrid", {'axes.grid': False})
    # sns.despine()
    # sns.histplot(data=ddist(c2_test, c1_test), kde=True, color="sandybrown", ax=ax, fill=True, stat="density",
    #              label="initial_distances")
    # sns.histplot(data=ddist(c2_test, c1_test_new), kde=True, color="tomato", ax=ax, fill=True, stat="density",
    #              label="transformed")
    # ax.set_xlabel("$d(nm)$")  # , fontsize=11, labelpad=30)
    # ax.set_ylabel("$Density$")  # , fontsize=45, labelpad=30)
    # ax.legend()
    # plt.tight_layout()
    # plt.savefig("../Exo70-Sec5/input/dsit.png")
    #
    # # CALCULATE TRE
    # tre_ = c1_test_new - c2_test
    # diff_x, diff_y = tre_[:, 0], tre_[:, 1]
    # df = pd.DataFrame({"x_W2": c2_test[:, 0],
    #                    "y_W2": c2_test[:, 1],
    #                    "x_W1": c1_test[:, 0],
    #                    "y_W1": c1_test[:, 1],
    #                    "diff_x": diff_x * 64.5,
    #                    "diff_y": diff_y * 64.5})
    #
    # # Gaussian fit to x and y data
    # data_x, data_y = df.diff_x.to_list(), df.diff_y.to_list()
    # mu_x, sigma_x = stats.norm.fit(df.diff_x.to_list())
    # mu_y, sigma_y = stats.norm.fit(df.diff_y.to_list())
    #
    # # ---------#
    # # Plot the fit
    # # x offset
    # figx, ax = plt.subplots()
    # sns.histplot(data=data_x, kde=True, color="grey", ax=ax, fill=True, stat="density",
    #              label=f"$\mu_x$ = {round(mu_x, 3)}\n$\sigma_x$ = {round(sigma_x, 3)}\nn = {len(data_x)}")
    # plt.plot(np.arange(min(data_x), max(data_x), 1),
    #          stats.norm.pdf(np.arange(min(data_x), max(data_x), 1),
    #                         loc=mu_x, scale=sigma_x), c='r', label="Gauss fit", linestyle='--')
    # ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    # plt.tight_layout()
    # plt.savefig("../Exo70-Sec5/input/ddsit_x.png")
    # plt.clf()
    #
    # # y offset
    # figy, ax = plt.subplots()
    # sns.histplot(data=data_y, kde=True, color="grey", ax=ax, fill=True, stat="density",
    #              label=f"$\mu_y$ = {round(mu_y, 3)}\n$\sigma_y$ = {round(sigma_y, 3)}\nn = {len(data_y)}")
    # plt.plot(np.arange(min(data_y), max(data_y), 1),
    #          stats.norm.pdf(np.arange(min(data_y), max(data_y), 1),
    #                         loc=mu_y, scale=sigma_y), c='r', label="Gauss fit", linestyle='--')
    # ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    # plt.tight_layout()
    # plt.savefig("../Exo70-Sec5/input/ddist_y.png")
    # plt.clf()
    #
    # # ---------#
    # # TRE
    # mean_x = np.mean(df.diff_x.to_numpy())
    # mean_y = np.mean(df.diff_y.to_numpy())
    # tre = np.sqrt(mean_x ** 2 + mean_y ** 2)
    # print(f"mean_x =  {np.round(mean_x, 3)} nm \n")
    # print(f"mean_y =  {np.round(mean_y, 3)} nm \n")
    # print(f"TRE = {np.round(tre, 3)} nm")
    #
    # # X shift
    # ax1 = sns.scatterplot(data=df, x="x_W2", y="y_W2", hue="diff_x", palette="RdBu")
    # norm = plt.Normalize(df['diff_x'].min(), df['diff_x'].max())
    # sm = plt.cm.ScalarMappable(cmap="RdBu", norm=norm)
    # sm.set_array([])
    #
    # # Remove the legend and add a colorbar
    # ax1.get_legend().remove()
    # ax1.figure.colorbar(sm)
    # ax1.invert_yaxis()
    # plt.savefig("../Exo70-Sec5/input/offset_x.png")
    # plt.clf()
    #
    # # Y shift
    # ax2 = sns.scatterplot(data=df, x="x_W2", y="y_W2", hue="diff_y", palette="RdBu")
    # norm = plt.Normalize(df['diff_y'].min(), df['diff_y'].max())
    # sm = plt.cm.ScalarMappable(cmap="RdBu", norm=norm)
    # sm.set_array([])
    #
    # # Remove the legend and add a colorbar
    # ax2.get_legend().remove()
    # ax2.figure.colorbar(sm)
    # ax2.invert_yaxis()
    # plt.savefig("../Exo70-Sec5/input/offset_y.png")
    # plt.clf()

    sys.exit()

# END
