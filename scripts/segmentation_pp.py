#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Python functions to preprocess PICT images by cell segmentation and
spot sorting based on nearest distance to contour and closest
neighbour distance.

Cell segmentation is done using YeastSpotter software code
(https://github.com/alexxijielu/yeast_segmentation)

"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # silence warnings

import shutil
import logging
import sys
import time
import glob

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from skimage.io import imread, imsave
import plotly.express as px
import plotly.graph_objects as go
from mrcnn.my_inference import predict_images
from mrcnn.preprocess_images import preprocess_images
from mrcnn.convert_to_image import convert_to_image, convert_to_imagej
from silence_tensorflow import silence_tensorflow

silence_tensorflow()  # Silence Tensorflow WARNINGS

# ======================
# SEGMENTATION FUNCTIONS
# ======================


def segment_yeast(segment_dir, images_dir, scale_factor, rescale, verbose=False, save_masks=True, output_imagej=False,
                  save_preprocessed=False, save_compressed=False):
    """
    Method to segment yeast cells using YeastSpotter software
    (https://github.com/alexxijielu/yeast_segmentation)
    """
    if segment_dir != '' and not os.path.isdir(segment_dir):
        os.mkdir(segment_dir)

    if os.path.isdir(segment_dir):
        if len(os.listdir(segment_dir)) > 0:
            logging.error("ERROR: Make sure that the output directory to save masks is empty.")
        else:
            preprocessed_image_directory = segment_dir + "preprocessed_images/"
            preprocessed_image_list = segment_dir + "preprocessed_images_list.csv"
            rle_file = segment_dir + "compressed_masks.csv"
            output_mask_directory = segment_dir + "masks/"
            output_imagej_directory = segment_dir + "imagej/"

            # Preprocess the images
            if verbose:
                logging.info("\nPreprocessing your images...")
            preprocess_images(images_dir,
                              preprocessed_image_directory,
                              preprocessed_image_list,
                              verbose=verbose)

            if verbose:
                logging.info("\nRunning your images through the neural network...")
            predict_images(preprocessed_image_directory,
                           preprocessed_image_list,
                           rle_file,
                           rescale=rescale,
                           scale_factor=scale_factor,
                           verbose=verbose)

            if save_masks:
                if verbose:
                    logging.info("\nSaving the masks...")

                if output_imagej:
                    convert_to_image(rle_file,
                                     output_mask_directory,
                                     preprocessed_image_list,
                                     rescale=rescale,
                                     scale_factor=scale_factor,
                                     verbose=verbose)

                    convert_to_imagej(output_mask_directory,
                                      output_imagej_directory)
                else:
                    convert_to_image(rle_file,
                                     output_mask_directory,
                                     preprocessed_image_list,
                                     rescale=rescale,
                                     scale_factor=scale_factor,
                                     verbose=verbose)

            os.remove(preprocessed_image_list)

            if not save_preprocessed:
                shutil.rmtree(preprocessed_image_directory)

            if not save_compressed:
                os.remove(rle_file)

            if not save_masks:
                shutil.rmtree(output_mask_directory)


def read_csv_2(file):
    """
    Function to read multiple csv (input data)
    """
    channel = file.split("/")[-1].split(".")[0].split("_")[3]
    df = pd.read_csv(file, sep="\t")  # sep="\t"
    if len(df) != 0:
        df.loc[:, "channel"] = channel
        return df


def calculate_distances(df_1, df_2, px_size):
    """
    Calculate distances (in nm) between coloc. spots
    """
    return np.sqrt(
        (df_1.x.to_numpy() - df_2.x.to_numpy()) ** 2 + (df_1.y.to_numpy() - df_2.y.to_numpy()) ** 2) * px_size


def get_label_coords(contour_label):
    """
    Method to get contour coordinates of labeled segmented cells
    Parameters
    ----------
    contour_label: ndimage with labeled contours

    Returns
    -------
    dictionary with label --> coordinates
    """
    unique_labels = set(contour_label[contour_label != 0].flatten())
    dict_cell_labels = dict()
    for label in unique_labels:
        label_coords = np.where(contour_label == label)
        dict_cell_labels.setdefault(label, np.array(list(zip(label_coords[0], label_coords[1]))))
    return dict_cell_labels


def clean_mother_daugther(segment_dir, img_num, contour_image, labels_image):
    """
    Method to avoid contour lines between mother and
    daughter cells.
    Parameters
    ----------
    img_num: image number ("01", "02", ...)
    contour_image: ndimage with cells contours (contour is 1, background is 0)
    labels_image: ndimage with labeled cells (each cell has a unique label)
    """
    # contour according to labels
    contour_label = contour_image * labels_image

    # Clean labels that are too close (> 2 px)
    # Group coordinates per cell (according to labels) in a dictionary
    dict_cell_labels = get_label_coords(contour_label)

    # Calculate distances between labels of different cells
    threshold = 2.5  # in pixels, to remove contour between two cells
    contour_to_modify = np.copy(contour_label)
    for label_1 in dict_cell_labels.keys():
        for label_2 in dict_cell_labels.keys():
            if label_1 != label_2:  # avoid calculating distances within a cell
                # Iterate over contour coords of cell-label_1 against all coords of cell-label_2
                for coord in dict_cell_labels[label_1]:
                    distances_to_coord = np.linalg.norm(coord - dict_cell_labels[label_2], axis=1)
                    # for coords fulfilling condition, replace values by a zero
                    close_labels = np.argwhere(distances_to_coord <= threshold)
                    if len(close_labels) > 0:
                        x1, x2 = coord[0], coord[1]
                        contour_to_modify[x1, x2] = 0
                        # for coords of cell-label_2 go to dict to get coordinates
                        for c in close_labels:
                            x = dict_cell_labels[label_2][c][0][0]
                            y = dict_cell_labels[label_2][c][0][1]
                            contour_to_modify[x, y] = 0
    imsave(segment_dir + "masks/contour_mod_{}.tif".format(img_num), contour_to_modify, plugin="tifffile",
           check_contrast=False)


def distance_to_contour(df_spots, contour_coordinates):
    """
    Method to get the closest distance from each spot to the cell contour
    Parameters
    ----------
    contour_coordinates: coordinates of cell contour
    df_spots: dataframe with  (x,y) coordinates

    Returns
    -------
    list of ditances
    """
    # Calculate min distance to contour and contour coordinate
    distances_cont = list()
    for coord in df_spots.to_numpy():
        spot_distances = np.linalg.norm(coord - contour_coordinates, axis=1)
        min_distance = spot_distances.min()
        contour_coord = tuple(contour_coordinates[spot_distances.argmin()])
        distances_cont.append((min_distance, contour_coord))
    return distances_cont


def distance_to_neigh(df_spots):
    """
    Method to get the closest neighbour for each spot to determine
    isolated spots
    Parameters
    ----------
    df_spots

    Returns
    -------

    """
    # Calculate min distance to contour and contour coordinate
    distances_neigh = list()
    for coord in df_spots.to_numpy():
        d_neigh = np.linalg.norm(coord - df_spots.to_numpy(), axis=1)
        # Check if more than 2 spots on the image
        if len(df_spots.to_numpy()) > 2:
            # check first 2 min distances (the min is 0 because its against the same spot, the closest spots
            # corresponds to the 2nd min dist)
            min_indexes = np.argpartition(d_neigh, 2)  # needs more than 2 elements in the list
            closest_neigh_dist = d_neigh[min_indexes[:2][1]]  # get the second min distance
            closest_neigh_idx = np.where(d_neigh == closest_neigh_dist)
            closest_neigh_spot = tuple(df_spots.to_numpy()[closest_neigh_idx][0])
            distances_neigh.append((closest_neigh_dist, closest_neigh_spot))
        elif len(df_spots.to_numpy()) == 2:
            closest_neigh_dist = d_neigh[1]
            closest_neigh_idx = np.where(d_neigh == closest_neigh_dist)
            closest_neigh_spot = tuple(df_spots.to_numpy()[closest_neigh_idx][0])
            distances_neigh.append((closest_neigh_dist, closest_neigh_spot))
    return distances_neigh


def sort_by_distances(spots_df, contour_coords, cont_cutoff=10, neigh_cutoff=10):
    """
    Method to sort spots based on distance to contour and distance to neighbour
    Parameters
    ----------
    neigh_cutoff
    cont_cutoff
    spots_df
    contour_coords

    Returns
    -------

    """
    # Get distance to contour and closes neigh
    sub_df = spots_df.loc[:, ["x", "y"]]
    contour_distances = distance_to_contour(sub_df, contour_coords)  # Calculate min distance to contour
    neigh_distances = distance_to_neigh(sub_df)  # Calculate closest neighbour distance

    # Add distances and coords to dataframe
    sub_df.loc[:, "dist_cont"], cont_coord_list = list(zip(*contour_distances))
    sub_df.loc[:, "contour_x"], sub_df.loc[:, "contour_y"] = list(zip(*cont_coord_list))
    if len(neigh_distances) > 0:
        sub_df.loc[:, "dist_neigh"], neigh_coord_list = list(zip(*neigh_distances))
        sub_df.loc[:, "neigh_x"], sub_df.loc[:, "neigh_y"] = list(zip(*neigh_coord_list))

    ########################################################################
    # Spot selection based on distance to contour and distance to neighbours
    ########################################################################
    # Label dataset with a "Selected" column
    if "dist_cont" in sub_df.columns and "dist_neigh" in sub_df.columns:
        sub_df.loc[:, 'selected'] = np.where((sub_df["dist_cont"] <= cont_cutoff) &
                                             (sub_df["dist_neigh"] > neigh_cutoff), "sel", "non-sel")
    elif "dist_cont" in sub_df.columns:
        sub_df.loc[:, 'selected'] = np.where((sub_df["dist_cont"] <= cont_cutoff), "sel", "non-sel")
    selection_df = spots_df.loc[(sub_df['selected'] == "sel")]
    # write to log percentage of selection
    num_selected = len(selection_df)
    percent_sel = num_selected * 100 / len(sub_df)

    return selection_df, sub_df, percent_sel


def save_html_figure(path_to_save, spots_df, img_num, img_contour_lab, ch_name="W1"):
    """
    Display selected and non-selected spots in an interactive image
    and save image in html format

    Parameters
    ----------
    ch_name: channel name: "W1" or "W2"
    spots_df: dataframe with spots coordinates for a given image
    img_num: image number ("01", "02", ...)
    img_contour_lab: binary image (ndarray) with contour as 1 and bgn as 0
    path_to_save: path to save image
    """
    if not os.path.exists(path_to_save):
        os.mkdir(path_to_save)
    if not os.path.exists(path_to_save + "pp_segmented/"):
        os.mkdir(path_to_save + "pp_segmented/")

    selected = spots_df[spots_df["selected"] == "sel"]
    non_selected = spots_df[spots_df["selected"] == "non-sel"]
    percent_sel = round(len(selected) * 100 / (len(selected) + len(non_selected)), 3)

    # Create figure with lines to closest contour and closest neighbour
    foo_note = "<br>Number of Selected spots: {} / {} (Percentage = {} %)<br><br>".format(len(selected),
                                                                                          len(selected) + len(
                                                                                              non_selected),
                                                                                          percent_sel)
    img_contour = (img_contour_lab > 0).astype("uint8")
    fig_label_cont = px.imshow(img_contour, color_continuous_scale='gray',
                               title="<b>Image {} {}</b><br>{}".format(img_num, ch_name, foo_note))
    fig_label_cont.update_layout(coloraxis_showscale=False)  # to hide color bar
    fig_label_cont.update_layout(coloraxis_showscale=False)

    # Plot spots with custom hover information
    if "dist_neigh" in selected.columns:
        custom_data = np.stack(([selected["dist_cont"], selected["dist_neigh"]]), axis=1)
    else:
        custom_data = selected["dist_cont"]
    fig_label_cont.add_scatter(x=selected["y"], y=selected["x"],
                               mode="markers",
                               marker=dict(color="green", size=7),
                               name="selected",
                               customdata=custom_data,
                               hovertemplate=
                               '<b>x: %{x: }</b><br>'
                               '<b>y: %{y: } <b><br>'
                               '<b>dist_cont: %{customdata[0]: }<b><br>'
                               '<b>dist_neigh: %{customdata[1]: }<b><br>')

    fig_label_cont.add_scatter(x=non_selected["y"], y=non_selected["x"],
                               mode="markers",
                               marker=dict(color="red"),
                               name="non-selected",
                               customdata=custom_data,
                               hovertemplate=
                               '<b>x: %{x: }</b><br>'
                               '<b>y: %{y: } <b><br>'
                               '<b>dist_cont: %{customdata[0]: }<b><br>'
                               '<b>dist_neigh: %{customdata[1]: }<b><br>')

    # Plot nearest contour spot
    fig_label_cont.add_scatter(x=spots_df["contour_y"], y=spots_df["contour_x"],
                               mode="markers", name="contour")

    # Plot blue lines from spots to its corresponding contour
    for index, row in spots_df.iterrows():
        fig_label_cont.add_trace(
            go.Scatter(
                x=[row["y"], row["contour_y"]],
                y=[row["x"], row["contour_x"]],
                mode="lines",
                line=go.scatter.Line(color="blue"),
                legendgroup="contour",
                name="{}".format(row["dist_cont"]), showlegend=True))

    # Plot orange lines from spots to its corresponding closest neighbour
    if "dist_neigh" in selected.columns:
        for index, row in spots_df.iterrows():
            fig_label_cont.add_trace(
                go.Scatter(
                    x=[row["y"], row["neigh_y"]],
                    y=[row["x"], row["neigh_x"]],
                    mode="lines",
                    legendgroup="neighbour",
                    line=go.scatter.Line(color="orange"),
                    name="{}".format(row["dist_neigh"]), showlegend=True))
    fig_label_cont.write_html(path_to_save + "pp_segmented/" + "image_{}_{}.html".format(img_num, ch_name))


def main_segmentation(segment_dir, images_dir, spots_dir, results_dir, figures_dir, scale_factor,
                      cont_cutoff, neigh_cutoff, px_size, rescale=False, verbose=False):
    """
    1) Main method to run segmentation preprocessing.
    """
    print("#############################\n"
          " Segmentation Pre-processing \n"
          "#############################\n")
    logging.info("\n\n###############################\n"
                 "Initializing Segmentation Analysis \n"
                 "###################################\n\n")
    ###############
    # SEGMENTATION
    ###############
    # Segment yeast cells if not segmented
    # Check if detected spots are present
    if not os.path.exists(images_dir) or len(os.listdir(images_dir)) == 0:
        sys.stderr.write('\nPICT-MODELLER-ERROR: Oh! You have to be a master for trying to segment phantom images! \n'
                         'You did not processed your raw images (option -pp), did you? If so, they disappeared :S\n'
                         'I can not segment an empty folder...Please, first process your images. \n'
                         'Thanks! ;)\n\n')
        sys.exit(1)
    if not os.path.exists(segment_dir):
        os.mkdir(segment_dir)
        segment_yeast(segment_dir, images_dir, scale_factor, rescale, verbose)  # saves contour images in output/masks/
        print("\n\nCell Segmentation Finished!\n\n")
    elif not len(glob.glob(images_dir + "image_*")) == len(glob.glob(segment_dir + "masks/image_*")):
        shutil.rmtree(segment_dir)
        segment_yeast(segment_dir, images_dir, scale_factor, rescale, verbose)  # saves contour images in output/masks/
        print("\n\nCell Segmentation Finished!\n\n")
    else:
        pass

    # Sort far-from-contour and/or not-isolated Spots
    # Check if detected spots
    if not os.path.exists(spots_dir) or len(os.listdir(spots_dir)) == 0:
        sys.stderr.write('\nPICT-MODELLER-ERROR: Oh wow! You still trying to hack me!!\n'
                         'You did not run spot detection (option -spt), did you? If so, I could not find any spot :S\n'
                         'I can not proceed if I do not have info about your spots... '
                         'Please, first process your images. \n'
                         'Thanks! ;)\n\n')
        sys.exit(1)

    percent_sel_total_W1 = list()
    percent_sel_total_W2 = list()
    percent_sel_total = list()
    total_data = 0
    total_selected = 0
    if os.path.exists(segment_dir + "masks/") and len(os.listdir(segment_dir + "masks/")) != 0:
        ###########################################################
        # Calculate distance to contour and closest neighbour distance
        ###########################################################
        for img_file in glob.glob(segment_dir + "masks/image_*"):
            start = time.time()
            image_number = img_file.split("/")[-1].split("_")[-1].split(".")[0]
            print("Processing image {} ...\n".format(image_number))
            # Read contour image and labeled image
            image_labels = imread(segment_dir + "masks/image_{}.tif".format(image_number)).astype(np.uint8)
            image_contour = imread(segment_dir + "masks/contour_image_{}.tif".format(image_number)) \
                .astype(np.uint8)

            # clean mother-bud cell barriers
            # Avoid doing this step if already done
            if not len(glob.glob(segment_dir + "masks/image_*")) == len(glob.glob(segment_dir +
                                                                                  "masks/contour_mod*")):
                print("\tCleaning mother-daugther cells...\n")
                clean_mother_daugther(segment_dir, image_number, image_contour, image_labels)  # generates an contour_mod image
                image_contour_mod = imread(segment_dir + "masks/contour_mod_{}.tif".format(image_number)) \
                    .astype(np.uint8)
            else:
                image_contour_mod = imread(segment_dir + "masks/contour_mod_{}.tif".format(image_number)) \
                    .astype(np.uint8)
            # Reading data from detected spots files in spots/
            # Check if file exists (meaning that trackpy could link spots on that image
            if os.path.exists(f'{spots_dir}detected_spot_{image_number}_W1_warped.csv') and os.path.exists(
                    f'{spots_dir}detected_spot_{image_number}_W2.csv'):
                spots_df_W1 = pd.read_csv(f'{spots_dir}detected_spot_{image_number}_W1_warped.csv',
                                          sep="\t", index_col=False)  # Spot coordinates W1
                spots_df_W2 = pd.read_csv(f'{spots_dir}detected_spot_{image_number}_W2.csv',
                                          sep="\t", index_col=False)  # Spot coordinates W2
                total_data += spots_df_W1.shape[0]

                # Add ID to each data point (spot)
                spots_df_W1.loc[:, "ID"] = list(range(1, spots_df_W1.shape[0] + 1))
                spots_df_W2.loc[:, "ID"] = list(range(1, spots_df_W2.shape[0] + 1))

                ###############################################
                # Sort by closest distance to contour and neigh
                ###############################################
                print("\tSorting spots...\n")
                cell_contour = np.where(image_contour_mod > 0)  # Group coordinates per cell (according to labels)
                cell_contour_coords = np.array(list(zip(cell_contour[0], cell_contour[1])))
                selection_df_W1, sub_df_W1, percent_sel_W1 = sort_by_distances(spots_df_W1,
                                                                               cell_contour_coords,
                                                                               cont_cutoff=cont_cutoff,
                                                                               neigh_cutoff=neigh_cutoff)
                selection_df_W2, sub_df_W2, percent_sel_W2 = sort_by_distances(spots_df_W2,
                                                                               cell_contour_coords,
                                                                               cont_cutoff=cont_cutoff,
                                                                               neigh_cutoff=neigh_cutoff)
                # Pair selected in W1 & W2
                selection_df_paired_W1 = selection_df_W1.loc[(selection_df_W1["ID"].isin(selection_df_W2["ID"]))]
                selection_df_paired_W2 = selection_df_W2.loc[(selection_df_W2["ID"].isin(selection_df_W1["ID"]))]
                # Assert shape W1 == shape W2
                assert set(selection_df_paired_W1.ID) == set(selection_df_paired_W2.ID)
                # update selected & non-selected values after pairing
                sub_df_W1["selected"] = np.where(~sub_df_W1["x"].isin(selection_df_paired_W1["x"]), "non-sel",
                                                 sub_df_W1["selected"])
                sub_df_W2["selected"] = np.where(~sub_df_W2["x"].isin(selection_df_paired_W2["x"]), "non-sel",
                                                 sub_df_W2["selected"])

                # write to log percentage of selection
                num_selected = selection_df_paired_W1.shape[0]
                percent_sel = num_selected * 100 / spots_df_W1.shape[0]
                logging.info("\nImage {} --> {:02} / {:02} "
                             "spots selected.. --> {} %".format(image_number, num_selected, len(spots_df_W1), percent_sel))
                total_selected += num_selected

                # Save df as csv: segmentation.csv
                if not os.path.exists(results_dir):
                    os.mkdir(results_dir)
                if not os.path.exists(results_dir + "segmentation/"):
                    os.mkdir(results_dir + "segmentation/")
                # If number of selected spots = 0, warn but not create file
                if num_selected > 0:
                    selection_df_paired_W1.to_csv(results_dir + "segmentation/" +
                                                  "detected_seg_{}_{}.csv".format(image_number, "W1"),
                                                  sep="\t", encoding="utf-8", header=True, index=False)
                    selection_df_paired_W2.to_csv(results_dir + "segmentation/" +
                                                  "detected_seg_{}_{}.csv".format(image_number, "W2"),
                                                  sep="\t", encoding="utf-8", header=True, index=False)
                else:
                    if os.path.exists(results_dir + f"{results_dir}segmentation/detected_seg_{image_number}_W1.csv"):
                        os.remove(results_dir + f"{results_dir}segmentation/detected_seg_{image_number}_W1.csv")
                    if os.path.exists(results_dir + f"{results_dir}segmentation/detected_seg_{image_number}_W2.csv"):
                        os.remove(results_dir + f"{results_dir}segmentation/detected_seg_{image_number}_W2.csv")
                # Create figure with lines to the closest contour and closest neighbour
                save_html_figure(figures_dir, sub_df_W1, image_number, image_contour_mod, ch_name="W1")
                save_html_figure(figures_dir, sub_df_W2, image_number, image_contour_mod, ch_name="W2")

                # Append percentages to list to write in report (log.txt)
                percent_sel_total_W1.append(percent_sel_W1)
                percent_sel_total_W2.append(percent_sel_W2)
                percent_sel_total.append(percent_sel)
                total_time = time.time() - start
                print("Image {} processed in {} s\n".format(image_number, round(total_time, 3)))

    logging.info("\n\nTotal Percent W1 --> {} %\n"
                 "Total Percent W2 --> {} %\n\n"
                 "Total Paired Percent --> {} % \n".format(sum(percent_sel_total_W1) / len(percent_sel_total_W1),
                                                           sum(percent_sel_total_W2) / len(percent_sel_total_W2),
                                                           sum(percent_sel_total) / len(percent_sel_total)))
    print("\n\nTotal Percent W1 --> {} %\n"
          "Total Percent W2 --> {} %\n\n"
          "Total Paired Percent --> {} % ({} spots) \n".format(sum(percent_sel_total_W1) / len(percent_sel_total_W1),
                                                               sum(percent_sel_total_W2) / len(percent_sel_total_W2),
                                                               sum(percent_sel_total) / len(percent_sel_total),
                                                               total_selected))
    # MEASURE DISTANCE DISTRIBUTION AFTER GAUSSIAN
    # Load data ensuring that W1 & W2 are paired
    df_W1 = pd.concat(map(read_csv_2, sorted(glob.glob(results_dir + "segmentation/detected_seg_*W1*"))),
                      ignore_index=True)
    df_W2 = pd.concat(map(read_csv_2, sorted(glob.glob(results_dir + "segmentation/detected_seg_*W2*"))),
                      ignore_index=True)
    initial_distances = np.loadtxt(results_dir + "distances_after_warping.csv")
    distances_seg = calculate_distances(df_W1, df_W2, px_size)
    np.savetxt(results_dir + "segmentation/seg_distances.csv", distances_seg, delimiter=",")

    # PLOT NEW DISTANCE DISTRIBUTION
    fig, ax = plt.subplots(figsize=(25, 15))
    sns.set(font_scale=1)
    sns.set_style("darkgrid")
    ax.set_title("Distances after KDE selection\n\n"
                 "mean detection = {} nm; stdev detection = {} nm; n = {}\n"
                 "mean seg = {} nm; stdev seg = {} nm; "
                 "n = {} \n".format(np.around(np.mean(initial_distances), 2),
                                    np.around(np.std(initial_distances), 2),
                                    len(initial_distances),
                                    np.around(np.mean(distances_seg), 2),
                                    np.around(np.std(distances_seg), 2),
                                    len(distances_seg)),
                 fontweight="bold", size=25)
    sns.histplot(data=initial_distances, kde=True, color="tab:grey", ax=ax, fill=True, stat="density")
    sns.histplot(data=distances_seg, kde=True, ax=ax, color="tab:red", fill=True, stat="density")
    ax.set_xlabel("Distances (nm) ", fontsize=45, labelpad=30)
    ax.set_ylabel("Density ", fontsize=45, labelpad=30)
    ax.tick_params(axis='x', labelsize=30)
    ax.tick_params(axis='y', labelsize=30)
    # ax.axvline(x=np.mean(initial_distances), color='sandybrown', ls='--', lw=2.5, alpha=0.8)
    # ax.axvline(x=np.mean(distances_seg), color='cornflowerblue', ls='--', lw=2.5, alpha=0.8)
    plt.grid()
    plt.savefig(figures_dir + "pp_segmented/" + "distances_after_segmentation.png")
    plt.clf()
    return total_data, total_selected


if __name__ == "__main__":
    print("Yeast Segmentation Functions :)\n")
    sys.exit(0)
