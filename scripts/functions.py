#!/usr/bin/python3.7
# coding=utf-8
"""
PyF2F functions to run the whole workflow.
"""
import trackpy as tp

from custom import *
from detect_beads import plot_mass, save_html_detection, save_html_selected_detection, piecewise_affine, ddist
from spot_detection_functions import detect_spots, link_particles
from segmentation_pp import *
from gaussian_fit import *
from kde import *
from outlier_rejection import *

os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")  # avoid cv2 and pyqt5 to be incompatible
logging.getLogger('matplotlib.font_manager').disabled = True

__author__ = "Altair C. Hernandez"
__copyright__ = 'Copyright 2023, PyF2F-Ruler'
__credits__ = ["J. Sebastian Ortiz", "Laura I. Betancur", "Radovan Dojcilovic", "Andrea Picco",
               "Marko Kaksonen", "Oriol Gallego"]
__version__ = "1.0"
__email__ = "altair.chinchilla@upf.edu"


def custom_read_csv(file):
    """
    Function to read multiple csv files (input data).

    :param file: path to csv file
    """
    channel = file.split("/")[-1].split(".")[0].split("_")[3]
    df = pd.read_csv(file, sep="\t")  # sep="\t"
    if len(df) != 0:
        df.loc[:, "channel"] = channel
        return df


def pp(pict_images_dir, path_to_save_pp, rbr_radius, mf_radius, verbose=False):
    """
    Pre-processing = Background Subtraction & Median Filter

    Method to pre-process raw images obtained from the PICT method
    (see Picco et al., 2017). Image pre-processing consists on:

    1) Removing the intensity noise from the image that falls under the size
    of a kernel defining a ball. The size of the kernel should be approximated
    to the largest object in the image.

    2) Removing the uneven intensities from the cellular environment by
    subtracting the median from the image.

    Parameters
    ----------
    :param pict_images_dir: path to raw .tif images
    :param path_to_save_pp: path to save pre-processed images.
    :param rbr_radius: rolling ball radius to define the kernel of 2*r
    :param mf_radius: median filter radius
    :param verbose: verbose


    """
    if verbose:
        print("\n#############################\n"
              "     Image Preprocessing \n"
              "#############################\n")
    # Check if detected spots are present
    print(pict_images_dir)
    if not os.path.exists(pict_images_dir) or len(os.listdir(pict_images_dir)) == 0:
        sys.stderr.write('\nPICT-MODELLER-ERROR: So fast!! You should drink a coffee first haha\n'
                         'Do you have PICT images? You sure? Go and check it!\n'
                         'Thanks! ;)\n\n')
        sys.exit(1)
    for file in glob.glob(pict_images_dir + "*.tif"):
        try:
            # BACKGROUND SUBTRACTION, MEDIAN FILTER
            image = BioImage(file).subtract_background(path_to_save_pp, rbr_radius)
            image.median_filter(path_to_save_pp, mf_radius)

        except FileNotFoundError as fnf:
            sys.stderr.write("{}\nWas not possible to do bead registration. Exit.".format(fnf))
            sys.exit()


def calculate_distances(df_1, df_2, px_size):
    """
    Calculate distances (in nm) between coloc. spots

    Parameters
    ----------
    :param df_1: Dataframe with XY coordinates of spots in channel 1
    :param df_2: Dataframe with XY coordinates of spots in channel 2
    :param px_size: pixel size of the camera.
    """
    return np.sqrt(
        (df_1.x.to_numpy() - df_2.x.to_numpy()) ** 2 + (df_1.y.to_numpy() - df_2.y.to_numpy()) ** 2) * px_size


def save_detected_distances(path_input, path_output, px_size):
    """
    Save the list of distances from detected spots

    Parameters
    ----------
    :param path_input: path to input spot detection csv files.
    :param path_output: path to save output txt file.
    :param px_size: px size of the camera.
    """
    # Save distances after spot location in results/trackpy.csv
    data_W1 = pd.concat(map(custom_kde_read_csv, sorted(glob.glob(path_input + "detected_*W1.csv"))),
                        ignore_index=True)
    data_W2 = pd.concat(map(custom_kde_read_csv, sorted(glob.glob(path_input + "detected_*W2.csv"))),
                        ignore_index=True)
    # Calculate distances
    distances = calculate_distances(data_W1, data_W2, px_size=px_size)
    # Save initial distances in results directory/spot_detection
    if not os.path.isdir(path_output + "spot_detection/"):
        os.mkdir(path_output + "spot_detection/")
    np.savetxt(path_output + "spot_detection/initial_distances.csv", distances, delimiter=",")
    return distances


def plot_distance_distribution(figures_dir, distances_array, save_name):
    """
    Plot distribution of distances and save in "figures_dir"  as "save_name".png

    Parameters
    ----------
    :param figures_dir: location to plot figures.
    :param distances_array: array of distances.
    :param save_name: name of file to be saved as PNG image
    """
    fig, ax = plt.subplots(figsize=(25, 20))
    sns.set(font_scale=2)
    ax.set_title("\nInitial Distances BEFORE Warping\n\n"
                 "mean = {} nm; stdev = {} nm\n".format(np.around(np.mean(distances_array), 2),
                                                        np.around(np.std(distances_array), 2),
                                                        fontweight="bold", size=35))
    sns.histplot(data=distances_array, kde=True, ax=ax, fill=True)
    ax.axvline(x=np.mean(distances_array), color='red', ls='--', lw=2.5, alpha=0.1)
    ax.set_xlabel("$Distances \ (nm) $", fontsize=25, labelpad=30)
    ax.set_ylabel("$Count $", fontsize=25, labelpad=30)
    ax.tick_params(axis='x', labelsize=30)
    ax.tick_params(axis='y', labelsize=30)
    if not os.path.isdir(figures_dir):
        os.mkdir(figures_dir)
    if not os.path.isdir(figures_dir + "spot_detection/"):
        os.mkdir(figures_dir + "spot_detection/")
    plt.savefig(figures_dir + f"spot_detection/{save_name}.png")


def plot_trackpy_spot_detection(spots_dir, figures_dir):
    """
    Plot additional information of the detected spots in the two channels
    such as the mass count, eccentrity, size, mass vx size.

     Parameters
    ----------
    :param spots_dir: where the W1 coordinates to warp are located.
    :param figures_dir: location to plot figures.
    """
    data = pd.concat(map(custom_read_csv, sorted(glob.glob(spots_dir + "detected*.csv"))),
                     ignore_index=True)
    data.mass = data.mass / 1000  # Just for visualization purposes
    sns.set(font_scale=3)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, constrained_layout=True, figsize=(30, 30))
    ax1.set_ylabel("Count", labelpad=30)
    ax1.set_xlabel("mass (I · 10³)", labelpad=30)
    ax2.set_ylabel("Count", labelpad=30)
    ax2.set_xlabel("size", labelpad=30)
    ax3.set_ylabel("Count", labelpad=30)
    ax3.set_xlabel("signal", labelpad=30)
    ax4.set_ylabel("Count", labelpad=30)
    ax4.set_xlabel("ecc", labelpad=30)
    sns.histplot(data=data, x="mass", kde=True, ax=ax1, hue="channel")
    ax1.axvline(x=data.mass.mean(), color='red', ls='--', lw=2.5, alpha=0.3)
    sns.histplot(data=data, x="size", kde=True, ax=ax2, hue="channel")
    ax2.axvline(x=data["size"].mean(), color='red', ls='--', lw=2.5, alpha=0.3)
    sns.histplot(data=data, x="signal", kde=True, ax=ax3, hue="channel")
    ax3.axvline(x=data.signal.mean(), color='red', ls='--', lw=2.5, alpha=0.3)
    sns.histplot(data=data, x="ecc", kde=True, ax=ax4, hue="channel")
    ax4.axvline(x=data.ecc.mean(), color='red', ls='--', lw=2.5, alpha=0.3)
    plt.savefig(figures_dir + "spot_detection/" + "spots_features.png")
    plt.clf()

    # Plotting mass vs ecc and mass vs signal
    fig, (ax5, ax6) = plt.subplots(2, 1, constrained_layout=True, figsize=(15, 15))
    tp.mass_size(data, ax=ax5)  # convenience function -- just plots size vs. mass
    tp.mass_ecc(data, ax=ax6)  # convenience function -- just plots ecc vs. mass
    plt.savefig(figures_dir + "spot_detection/" + "spots_features_2.png")
    plt.clf()

    # 3D plot mass - ecc - signal
    fig3 = px.scatter_3d(data, x='mass', y='ecc', z='signal',
                         color='channel')
    fig3.write_html(figures_dir + "spot_detection/" + "mass_ecc_signal.html")

    # 3D plot mass - ecc - size
    fig4 = px.scatter_3d(data, x='mass', y='ecc', z='size',
                         color='channel')
    fig4.write_html(figures_dir + "spot_detection/" + "mass_ecc_size.html")


def spot_detection(images_dir, spots_dir, results_dir, figures_dir,
                   particle_diameter, percentile, max_displacement, px_size,
                   min_mass_cutoff=None, max_mass_cutoff=None, verbose=False,
                   mass_selection=False, test=False, dirty=False):
    """
    Method to run spot detection from pre-processed images.

    Parameters
    ----------
    :param images_dir: where the input images are located.
    :param spots_dir: where the W1 coordinates to warp are located.
    :param results_dir: location to save results.
    :param figures_dir: location to plot figures.
    :param particle_diameter: diameter in pixels of gaussian-like profiles to be detected
    :param percentile: select pixels brighter than this percentile (%).
    :param max_displacement: max separation in pixels to link spots between two channels.
    :param px_size: pixel size of the camera in nanometers
    :param min_mass_cutoff: deprecated, trackpy handles this selection.
    :param max_mass_cutoff: deprecated, trackpy handles this selection.
    :param verbose: print output of trackpy
    :param mass_selection: deprecated, trackpy handles this selection.
    :param test: test this function
    :param dirty: run with HTML outputs for each image (slow)

    Returns
    -------
    DataFrame with detected spots.

        DataFrame([x, y, mass, size, ecc, signal]);

        where mass means total integrated brightness of the blob, size means the radius of gyration
        of its Gaussian-like profile, and ecc is its eccentricity (0 is circular).
    """
    tp.quiet()
    # Create directories to save output
    if not os.path.exists(figures_dir):
        os.mkdir(figures_dir)
    if not os.path.exists(figures_dir + "spot_detection/"):
        os.mkdir(figures_dir + "spot_detection/")
    if not os.path.exists(figures_dir + "spot_detection/detection/"):
        os.mkdir(figures_dir + "spot_detection/detection/")
    if not os.path.exists(figures_dir + "spot_detection/linked_spots/"):
        os.mkdir(figures_dir + "spot_detection/linked_spots/")
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    total_initial_W1 = 0
    total_initial_W2 = 0
    total_final = 0
    if os.path.exists(images_dir) and len(glob.glob(images_dir)) != 0:
        # Create path to Trackpy spots in the case it is not created
        if not os.path.isdir(spots_dir):
            os.mkdir(spots_dir)
        if verbose:
            print("##########################################\n"
                  " ---Running SPOT LOCATION AND TRACKING--- \n"
                  "##########################################\n")
        for img in sorted(glob.glob(images_dir + "imageMD*.tif")):
            path = "/".join(img.split("/")[:-1])
            img_name = img.split("/")[-1]
            ndimage = io.imread(path + "/imageMD_{}.tif".format(img_name.split(".")[0].split("_")[1]))
            if verbose:
                print("# IMAGE {} \n#".format(img_name))
            # SPOT DETECTION
            f_batch_det, num_W1, num_W2 = detect_spots(images_dir, img_name, particle_diameter,
                                                       percentile, max_mass_cutoff, min_mass_cutoff,
                                                       verbose, mass_selection=mass_selection)
            # Plot mass distribution and selected spots
            if mass_selection:
                plot_mass(figures_dir + "spot_detection/detection/", f_batch_det, img_name)
                if dirty:
                    save_html_selected_detection(figures_dir + "spot_detection/detection/", f_batch_det, img_name,
                                                 ndimage, percentile, min_mass_cutoff, max_mass_cutoff)
            else:
                if dirty:
                    save_html_detection(figures_dir + "spot_detection/detection/", f_batch_det, img_name, ndimage,
                                        percentile)
            total_initial_W1 += num_W1
            total_initial_W2 += num_W2
            if len(f_batch_det) != 0:
                # LINKING
                if verbose:
                    print("\nSpot Detection done\n\tAligning W1 and W2 files..\n")
                if mass_selection:
                    f_batch_det = f_batch_det.copy()[f_batch_det['selected'] == "sel"].drop(columns=["selected"])
                f_batch_link, t_only_paired, num_linked_spots = link_particles(f_batch_det, img_name, spots_dir,
                                                                               max_displacement,
                                                                               verbose, mass_selection)
                total_final += num_linked_spots
                if mass_selection:
                    if dirty:
                        # PLOT selected after linking the two channels
                        save_html_selected_detection(figures_dir + "spot_detection/linked_spots/", f_batch_link,
                                                     img_name,
                                                     ndimage, percentile, min_mass_cutoff, max_mass_cutoff)
                else:
                    if dirty:
                        save_html_detection(figures_dir + "spot_detection/linked_spots/", t_only_paired, img_name,
                                            ndimage,
                                            percentile)

        print("\nTotal Initial W1 Detected spots: {}\n"
              "\nTotal Initial W2 Detected spots: {}\n"
              "\nTotal Final Paired spots: {}\n".format(total_initial_W1, total_initial_W2, total_final))

        # WARNING
        if not test and total_final <= 100:
            # WARNING: if number of paired spots < 100, poor dataset or poor quality images.
            # The following analysis could give misleading - biased - non reliable results!
            sys.stderr.write('PICT-WARNING: Trackpy could pair less than 100 spots in your image dataset '
                             f'{"/".join(images_dir.split("/")[:3])}. '
                             'Probably due to a short dataset '
                             'or poor quality images :(\n'
                             'The following analysis could retrieve misleading - biased - non reliable results!\n '
                             '\t\tWe strongly recommend a minimum number of input images == 20.\n'
                             '\t\tPlease, review your input image dataset quality and run it again.\n\n\t'
                             'Good luck! :)\n\n')
            # sys.exit(1)

        # Save distances after spot location in results/spot_detection
        initial_distances = save_detected_distances(spots_dir, results_dir, px_size)
        plot_distance_distribution(figures_dir, initial_distances, save_name="initial_distances")

        # Plot histograms of spot properties (mass, size, signal, mass vs ecc)
        plot_trackpy_spot_detection(spots_dir, figures_dir)


def plot_links(images_dir, spots_dir, figures_dir):
    """
    Visual check of linking (pairing) of W2 and W1 spots.
    Also allows visualizing the new W1 (W1_warped) coordinate.

    Parameters
    ----------
    :param images_dir: where the input images are located.
    :param spots_dir: where the W1 coordinates to warp are located.
    :param figures_dir: location to plot figures.

    """
    for img in glob.glob(images_dir):
        img_name = img.split("/")[-1].split(".")[0]
        num = img_name.split("_")[-1]
        if os.path.exists(f'{spots_dir}detected_spot_{num}_W1_warped.csv') and os.path.exists(
                f'{spots_dir}detected_spot_{num}_W2.csv'):
            if img_name.startswith("imageMD"):
                ndimage = io.imread(img)[1, :, :]  # plot on W2
            else:
                ndimage = io.imread(img)
            coords_W1 = np.loadtxt(spots_dir + "detected_spot_{}_W1.csv".format(num), delimiter="\t", usecols=[0, 1],
                                   skiprows=1)
            coords_W1_warped = np.loadtxt(spots_dir + "detected_spot_{}_W1_warped.csv".format(num), delimiter="\t",
                                          usecols=[0, 1], skiprows=1)
            coords_W2 = np.loadtxt(spots_dir + "detected_spot_{}_W2.csv".format(num), delimiter="\t", usecols=[0, 1],
                                   skiprows=1)
            # Plot initial coordinates of W1 and W2 and a line indicating pairs
            fig = px.imshow(ndimage, color_continuous_scale='gray',
                            title="<b>Check pairing img {} in W2 image or segmented<br>".format(num))
            fig.update_layout(coloraxis_showscale=False)  # to hide color bar
            if coords_W1.ndim == 2 and coords_W1_warped.ndim == 2 and coords_W2.ndim == 2:
                fig.add_scatter(x=coords_W2[:, 1], y=coords_W2[:, 0],
                                mode="markers",
                                marker_symbol="circle-dot",
                                marker=dict(color="green", size=10,
                                            line=dict(width=2,
                                                      color='green')),
                                name="W2",
                                opacity=0.7)
                fig.add_scatter(x=coords_W1[:, 1], y=coords_W1[:, 0],
                                mode="markers",
                                marker_symbol="square-dot",
                                marker=dict(color="red", size=10,
                                            line=dict(width=2,
                                                      color='red')),
                                name="W1",
                                opacity=0.7)

                # Draw lines between W2 - W1 before warping to see link
                for i in range(len(coords_W1)):
                    y1, y1_warped, y2 = coords_W1[i][1], coords_W1_warped[i][1], coords_W2[i][1]
                    x1, x1_warped, x2 = coords_W1[i][0], coords_W1_warped[i][0], coords_W2[i][0]
                    fig.add_trace(go.Scatter(x=[y1, y2], y=[x1, x2],
                                             mode="lines",
                                             line=go.scatter.Line(color="black"),
                                             name="{}".format(i), showlegend=False))
                    fig.add_trace(go.Scatter(x=[y1_warped, y2], y=[x1_warped, x2],
                                             mode="lines",
                                             line=go.scatter.Line(color="purple"),
                                             name="{}".format(i), showlegend=False))

                # PLOT WARPED COORDS IN IMAGE AFTER TRANSFORMATION
                fig.add_scatter(x=coords_W1_warped[:, 1], y=coords_W1_warped[:, 0],
                                mode="markers",
                                marker_symbol="x-dot",
                                marker=dict(color="orange", size=10,
                                            line=dict(width=2,
                                                      color='orange')),
                                name="W1_warped",
                                opacity=0.7)
            if not os.path.exists(figures_dir + "check_link/"):
                os.mkdir(figures_dir + "check_link/")
            fig.write_html(figures_dir + "check_link/check_link_{}.html".format(num))


def global_warping(beads_dir, spots_dir, figures_dir, results_dir, pixel_size):
    """
    Warping(chromatic aberration correction) of centroid coordinates from
    W1/red channel spots. It uses the centroid of BEAD IMAGES 4xFOV
    (W2.tif --> ref_centroid; W1.tif --> mov_centroid) and the transformation
    matrix from BEADS correction to correct all experimental W1 coordinates
    (W1 --> W1_warped).

    Parameters
    ----------
    :param beads_dir: path to coordinate files for REF beads
    :param spots_dir: where the W1 coordinates to warp are located.
    :param figures_dir: location to plot figures.
    :param results_dir: location to save results.
    :param pixel_size: pixel size of the camera in nm
    """
    ref_x, ref_y = list(), list()
    mov_x, mov_y = list(), list()
    warped_coords_x, warped_coords_y = list(), list()
    # REGISTRATION & TRANSFORMATION
    for file in sorted(glob.glob(spots_dir + "detected*_W2.csv")):
        path = "/".join(file.split("/")[:-1])
        num = file.split("/")[-1].split("_")[2]
        W2_data = pd.read_csv(path + "/detected_spot_{}_W2.csv".format(num), sep="\t")
        W1_data = pd.read_csv(path + "/detected_spot_{}_W1.csv".format(num), sep="\t")
        # Check if there are images without linked spots, warn about it and
        # erase file detected*_W1.csv and detected*_W2.csv
        if W1_data.shape[0] > 0 and W2_data.shape[0] > 0:
            ref_ = np.asarray(list(zip(W2_data.x, W2_data.y)))
            mov_ = np.asarray(list(zip(W1_data.x, W1_data.y)))
            # Load transformation matrix, and
            transformation = np.load(beads_dir + "transform.npy")
            ref_centroid = np.load(beads_dir + "ref_centroid.npy")
            mov_centroid = np.load(beads_dir + "mov_centroid.npy")
            # TRANSFORMATION
            new_coords = np.empty_like(ref_)
            for i in range(len(ref_)):
                new_coords[i] = ref_centroid + np.dot(transformation, mov_[i] - mov_centroid)

            W1_warped_data = W1_data.copy()
            W1_warped_data.loc[:, ("x", "y")] = new_coords
            W1_warped_data.to_csv(spots_dir + "/detected_spot_{}_W1_warped.csv".format(num), sep="\t",
                                  encoding="utf-8", header=True, index=False)
            ref_x += W2_data.x.tolist()
            ref_y += W2_data.y.tolist()
            mov_x += W1_data.x.tolist()
            mov_y += W1_data.y.tolist()
            warped_coords_x += new_coords[:, 0].tolist()
            warped_coords_y += new_coords[:, 1].tolist()
        else:
            print(f'\n\t0 spots found in {file}! --> can not do warping --> '
                  f'removing file {file} and its partner W1\n\n')
            time.sleep(3)
            os.remove(f'{path}/detected_spot_{num}_W1.csv')
            os.remove(file)
    # Distance measurements BEFORE & AFTER WARPING
    ref = np.asarray(list(zip(ref_x, ref_y)))
    mov = np.asarray(list(zip(mov_x, mov_y)))
    warped_coords = np.asarray(list(zip(warped_coords_x, warped_coords_y)))
    original_distances = np.sqrt(
        (ref[:, 0] - mov[:, 0]) ** 2 + (ref[:, 1] - mov[:, 1]) ** 2) * pixel_size
    new_distances = np.sqrt(
        (ref[:, 0] - warped_coords[:, 0]) ** 2 + (ref[:, 1] - warped_coords[:, 1]) ** 2) * pixel_size
    np.savetxt(results_dir + "distances_after_warping.csv", new_distances, delimiter=",")
    # PLOT BEFORE vs AFTER WARPING
    fig, ax = plt.subplots(figsize=(25, 15))
    sns.set(font_scale=3)
    ax.set_title("Beads Distances AFTER Warping\n\n"
                 "mean before = {} nm; stdev before = {} nm\n"
                 "mean after = {} nm; stdev after = {} nm \n".format(np.around(np.mean(original_distances), 2),
                                                                     np.around(np.std(original_distances), 2),
                                                                     np.around(np.mean(new_distances), 2),
                                                                     np.around(np.std(new_distances)), 2),
                 fontweight="bold", size=25)
    sns.histplot(data=original_distances, kde=True, color="sandybrown", ax=ax, fill=True)
    sns.histplot(data=new_distances, kde=True, ax=ax, color="cornflowerblue", fill=True)
    ax.set_xlabel("$Distances (nm) $", fontsize=45, labelpad=30)
    ax.set_ylabel("$Count $", fontsize=45, labelpad=30)
    ax.axvline(x=np.mean(original_distances), color='sandybrown', ls='--', lw=2.5, alpha=0.8)
    ax.axvline(x=np.mean(new_distances), color='cornflowerblue', ls='--', lw=2.5, alpha=0.8)
    plt.savefig(figures_dir + "spot_detection/distances_after_warping.png")


def local_warping(beads_dir, spots_dir, figures_dir, results_dir, pixel_size):
    """
    Loca warping(chromatic aberration correction) of centroid coordinates from
    W1/red channel spots. It uses a piecewise affine transformation based on a
    nearest neighbour search. centroid of BEAD IMAGES 4xFOV
    (W2.tif --> ref_centroid; W1.tif --> mov_centroid) and the transformation
    matrix from BEADS correction to correct all experimental W1 coordinates
    (W1 --> W1_warped).

    Parameters
    ----------
    :param beads_dir: path to coordinate files for REF beads
    :param spots_dir: where the W1 coordinates to warp are located.
    :param figures_dir: location to plot figures.
    :param results_dir: location to save results.
    :param pixel_size: pixel size of the camera in nm
    """
    search_range = 2000 / pixel_size  # search range from X,Y coordinate to be transformed
    min_fiducials = 10                # min number of fiducials in the search range
    # Get coordinates from beads for registration
    W1_ref = np.loadtxt(beads_dir + "/coords_W1.csv", delimiter=",")
    W2_ref = np.loadtxt(beads_dir + "/coords_W2.csv", delimiter=",")

    # Warp W1 coordinates locally using the REF beads coordinates
    # to calculate the registration map
    for file in sorted(glob.glob(spots_dir + "detected*_W2.csv")):
        path = "/".join(file.split("/")[:-1])
        num = file.split("/")[-1].split("_")[2]
        W2_data = pd.read_csv(path + "/detected_spot_{}_W2.csv".format(num), sep="\t")
        W1_data = pd.read_csv(path + "/detected_spot_{}_W1.csv".format(num), sep="\t")
        # Check if there are images without linked spots, warn about it and
        # erase file detected*_W1.csv and detected*_W2.csv
        if W1_data.shape[0] > 0 and W2_data.shape[0] > 0:
            # Get X,Y coords for W1 and W2
            W1_coords = np.asarray(list(zip(W1_data.x, W1_data.y)))
            W2_coords = np.asarray(list(zip(W2_data.x, W2_data.y)))
            W1_warped_coords = np.empty_like(W1_coords)
            for i in range(len(W1_coords)):
                W1_warped_coords[i] = piecewise_affine(W1_coords[i], W1_ref, W2_ref, search_range=search_range,
                                                       min_candidates=min_fiducials)
            # Save W1_warped coords
            W1_warped_data = W1_data.copy()
            W1_warped_data.loc[:, ("x", "y")] = W1_warped_coords
            W1_warped_data.to_csv(spots_dir + "/detected_spot_{}_W1_warped.csv".format(num), sep="\t",
                                  encoding="utf-8", header=True, index=False)
        else:
            print(f'\n\t0 spots found in {file}! --> can not do warping --> '
                  f'removing file {file} and its partner W1\n\n')
            time.sleep(3)
            os.remove(f'{path}/detected_spot_{num}_W1.csv')
            os.remove(file)

    # Distance distribution BEFORE vs AFTER warping
    # Load sample coords for W1 and W2 channel
    data_W1 = pd.concat(map(custom_kde_read_csv, sorted(glob.glob(spots_dir + "detected_*W1.csv"))), ignore_index=True)
    data_W2 = pd.concat(map(custom_kde_read_csv, sorted(glob.glob(spots_dir + "detected_*W2.csv"))), ignore_index=True)
    data_W1_warped = pd.concat(map(custom_kde_read_csv, sorted(glob.glob(spots_dir + "detected_*W1_warped.csv"))),
                               ignore_index=True)
    # get X,Y coords for W1 and W2
    W1_coords = np.asarray(list(zip(data_W1.x, data_W1.y)))
    W2_coords = np.asarray(list(zip(data_W2.x, data_W2.y)))
    W1_warped_coords = np.asarray(list(zip(data_W1_warped.x, data_W1_warped.y)))
    # PLOT
    fig, ax = plt.subplots()
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.despine()
    sns.histplot(data=ddist(W2_coords, W1_coords, px_size=pixel_size), kde=True, color="sandybrown", ax=ax, fill=True,
                 stat="density", label="initial_distances")
    sns.histplot(data=ddist(W2_coords, W1_warped_coords, px_size=pixel_size), kde=True, color="tomato", ax=ax, fill=True,
                 stat="density", label="transformed")
    ax.set_xlabel("$d \ (nm)$")  # , fontsize=11, labelpad=30)
    ax.set_ylabel("$Density$")  # , fontsize=45, labelpad=30)
    ax.legend()
    plt.savefig(figures_dir + "spot_detection/distances_after_warping.png")
    # plt.show()
    # Save warped coords in results/
    np.savetxt(results_dir + "distances_after_warping.csv", ddist(W2_coords, W1_warped_coords, px_size=pixel_size),
               delimiter=",")
























