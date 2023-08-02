"""
Python functions for spot location using Trackpy modules
"""
import trackpy as tp
import pims
from detect_beads import select_mass_cdf
import numpy as np


def detect_spots(path_to_folder, bgn_img, diameter, percentile,
                 max_mass_cutoff=None, min_mass_cutoff=None, verbose=False, mass_selection=False):
    """
    Method for spot detection from already Background Subtracted images.
    Parameters
    ----------
    bgn_img: name of the background subtracted image
    path_to_folder
    diameter: float. Diameter in pixels of the spots to search.
    percentile: float. Percentile (%) that determines which bright pixels are accepted as "spots".
    max_mass_cutoff: float. Reject the upper right of this cutoff (in tant per 1)
    min_mass_cutoff: float. Reject the upper left  of this cutoff (in tant per 1)
    verbose
    mass_selection

    Returns
    -------
    DataFrame with detected spots.

        DataFrame([x, y, mass, size, ecc, signal]);

        where mass means total integrated brightness of the blob, size means the radius of gyration
        of its Gaussian-like profile, and ecc is its eccentricity (0 is circular).

    """
    # Load frames using PIMS
    frames = pims.open(path_to_folder + bgn_img)
    img_num = "{}".format(bgn_img.split(".")[0].split("_")[-1]) # .replace("Pos", ""))  # image number (string)
    if verbose:
        print("\t\t{} frames in stack...\n".format(len(frames)))
        print("# SPOT DETECTION with Trackpy...\n\n\t# READING DATA...\n\n"
              "\t\tParticle diameter = {}\n"
              "\t\tPercentile = {}\n\n".format(diameter, percentile))

    f_batch = tp.batch(frames[:], diameter, percentile=percentile, engine='python')
    f_batch = f_batch.rename(columns={"x": "y", "y": "x"})  # change x,y order
    f_batch['size'] = f_batch['size'].apply(lambda x: x ** 2)  # remove sqrt from size formula
    f_batch["img"] = img_num
    if mass_selection:
        # Select spots with a cumulative density probability less than a threshold
        f_batch.loc[:, "ID"] = list(range(1, f_batch.shape[0] + 1))
        mass_selection = select_mass_cdf(f_batch.mass.to_numpy(), bins=100, max_mass_cutoff=max_mass_cutoff,
                                         min_mass_cutoff=min_mass_cutoff, verbose=verbose, debug=False)
        f_batch['selected'] = np.where(f_batch.mass.isin(mass_selection), "sel", "non-sel")
        frame_0, frame_1 = len(f_batch[((f_batch["frame"] == 0) & (f_batch["selected"] == "sel"))]), \
                           len(f_batch[((f_batch["frame"] == 1) & (f_batch["selected"] == "sel"))])
        print("\nInitial number of spots detected by trackpy: {}\n"
              "Number of spots discarded regarding mass: {}\n"
              "Final number of selected spots: {}\n\n".format(f_batch.shape[0],
                                                              len(f_batch[
                                                                      f_batch["selected"] ==
                                                                      "non-sel"]),
                                                              len(mass_selection)))
    else:
        frame_0, frame_1 = len(f_batch[f_batch["frame"] == 0]), len(f_batch[f_batch["frame"] == 1])
    if verbose:
        print("\t\t\tFrame 0: {} particles detected\n"
              "\t\t\tFrame 1: {} particles detected\n"
              "\t\t\tPossible paired particles: {}\n".format(frame_0, frame_1, min(frame_0, frame_1)))
    return f_batch, frame_0, frame_1


def link_particles(f_batch_df, img, path_to_output, maximum_displacement, verbose=False, mass_selection=False):
    """
    Recurse for linking already found particles into particle trajectories.
    Parameters
    ----------
    f_batch_df: pd.Dataframe. DataFrame([x, y, mass, size, ecc, signal])
    img: image name for saving file
    path_to_output: where to save the output.
    maximum_displacement: how much a particle may move between frames to be considered to link.
    verbose
    mass_selection

    Returns
    -------
    Nothing to return here.

    """
    img_num = img.split(".")[0].split("_")[-1]  # if is a stack it should be [3]
    out_traj_file = "detected_spot_{}".format(img_num)
    if verbose:
        print("\t# LINKING INTO PARTICLE TRAJECTORIES...\n\n"
              "\t\tSearch range: {}\n".format(maximum_displacement))
    t = tp.link(f_batch_df, maximum_displacement, pos_columns=["x", "y"])
    t_sort_particles = t.sort_values(by=["particle", "frame"])
    t_only_paired = t_sort_particles[t_sort_particles.duplicated("particle", keep=False)]
    if mass_selection:
        f_batch_df.loc[:, "selected"] = np.where(f_batch_df.ID.isin(t_only_paired.ID.tolist()), "sel",
                                                 "non-sel")
    # Split linked particles by frame
    t_filtered_W1 = t_only_paired[t_only_paired["frame"] == 0]
    t_filtered_W2 = t_only_paired[t_only_paired["frame"] == 1]
    if verbose:
        print("\t\tNumber of trajectories detected: {}".format(len(t_filtered_W1)))
    t_filtered_W1.to_csv(path_to_output + out_traj_file + "_W1.csv", sep="\t", encoding="utf-8",
                         header=True, index=False)
    t_filtered_W2.to_csv(path_to_output + out_traj_file + "_W2.csv", sep="\t", encoding="utf-8", header=True,
                         index=False)
    if verbose:
        print("\t\tSaving trajectories at {}.\n\n".format(path_to_output + out_traj_file))
    return f_batch_df, t_only_paired, t_filtered_W2.shape[0]


if __name__ == "__main__":
    print("Hi mate! This file only contains python functions for SPOT DETECTION using Trackpy."
          "You may want to use them, open it and have a look ;)")
