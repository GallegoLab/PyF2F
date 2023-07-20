#!/usr/bin/python3.7
# coding=utf-8
"""
######################################################
#################  RUN PYF2F-RULER #################
######################################################

BioImage Analysis pipeline to estimate distances between two fluorophores using the PICT method
(see Picco A. et al., 2017).
"""
from functions import *
from argparse import ArgumentParser
import matplotlib as mpl

mpl.rcParams.update({'figure.max_open_warning': 0})

__author__ = "Altair C. Hernandez"
__copyright__ = 'Copyright 2023, PyF2F-Ruler'
__credits__ = ["J. Sebastian Ortiz", "Laura I. Betancur", "Radovan Dojcilovic", "Andrea Picco",
               "Marko Kaksonen", "Oriol Gallego"]
__version__ = "1.0"
__email__ = "altair.chinchilla@upf.edu"

#########
# PARSER
#########

parser = ArgumentParser(prog='PyF2F-Ruler',
                        description='Pipeline to estimate the distance between a fluorophore fused to the termini '
                                    'of a protein complex and a reference fluorophore in the anchor in living yeast'
                                    '(see PICT method in Picco et al., 2017)')

parser.add_argument("-d", "--dataset",
                    dest="dataset",
                    action="store",
                    type=str,
                    required=True,
                    help="Name of the main directory where the dataset is located")

parser.add_argument('--px_size',
                    dest='px_size',
                    action="store",
                    type=float,
                    default=64.5,
                    help="Pixel size of the camera (nanometers)")

parser.add_argument('--rolling_ball_radius',
                    dest='rolling_ball_radius',
                    action="store",
                    type=int,
                    default=70,
                    help="Rolling Ball Radius (pixels)")

parser.add_argument('--median_filter',
                    dest='median_filter',
                    action="store",
                    type=int,
                    default=10,
                    help="Median Radius (pixels)")

parser.add_argument('--particle_diameter',
                    dest='particle_diameter',
                    action="store",
                    type=int,
                    default=11,
                    help="For spot detection. Must be an odd number")

parser.add_argument('--percentile',
                    dest='percentile',
                    action="store",
                    type=float,
                    default=99.7,
                    help="Percentile that determines which bright pixels are accepted as spots.")

parser.add_argument('--max_displacement',
                    dest='max_displacement',
                    action="store",
                    type=float,
                    default=2,
                    help="Median Radius (pixels)")

parser.add_argument('--local_transformation',
                    dest='local_transformation',
                    action="store_true",
                    help="Use local affine (piecewise) transformation instead of global affine.")

parser.add_argument('--contour_cutoff',
                    dest='contour_cutoff',
                    action="store",
                    type=int,
                    default=13,
                    help="Max distance to cell contour (pixels)")

parser.add_argument('--neigh_cutoff',
                    dest='neigh_cutoff',
                    action="store",
                    type=int,
                    default=9,
                    help="Max distance to closest neighbour (pixels)")

parser.add_argument('--kde_cutoff',
                    dest='kde_cutoff',
                    action="store",
                    type=float,
                    default=0.5,
                    help="Spots with this probability to be found in the population")

parser.add_argument('--gaussian_cutoff',
                    dest='gaussian_cutoff',
                    action="store",
                    type=float,
                    default=0.35,
                    help="Spots with this probability to be found in the population")

parser.add_argument('--mle_cutoff',
                    dest='mle_cutoff',
                    action="store",
                    type=float,
                    default=2 / 3,
                    help="In the MLE, tant per cent of the distribution assumed to be ok. "
                         "Outlier search in the right '1 - value' area of the distance distribution")

parser.add_argument('--reject_lower',
                    dest='reject_lower',
                    action="store",
                    type=float,
                    default=0,
                    help="In the MLE, reject selected values under this threshold")

parser.add_argument('--mu_ini',
                    dest='mu_ini',
                    action="store",
                    type=int,
                    default="auto",
                    help="Initial guess for mu search in the MLE")

parser.add_argument('--sigma_ini',
                    dest='sigma_ini',
                    action="store",
                    type=int,
                    default="auto",
                    help="Initial guess for sigma search in the MLE")

parser.add_argument('--bin_size',
                    dest='bin_size',
                    action="store",
                    type=float,
                    default=None,
                    help="Bin size for distance distribution plots")

parser.add_argument('--dirty',
                    dest='dirty',
                    action="store_true",
                    default=False,
                    help="Generates an html file to show the spots selected/rejected in each image for each "
                         "step of the process. Consumes more time, memory, and local space.")

parser.add_argument('--verbose',
                    dest='verbose',
                    action="store_true",
                    default=False,
                    help="Informs in the terminal what the program is doing at each step")

parser.add_argument('-o',
                    '--option',
                    dest='option',
                    action="store",
                    default="all",
                    help="Option to process:"
                         " 'all' (whole workflow),"
                         " 'beads' (bead registration),"
                         " 'pp' (preprocessing),"
                         " 'spt' (spot detection and linking),"
                         " 'warping' (transform XY spot coordinates using the beads registration map),"
                         " 'segment' (yeast segmentation),"
                         " 'kde' (2D Kernel Density Estimate),"
                         " 'gaussian' (gaussian fitting),"
                         " 'mle (outlier rejection using the MLE)'."
                         " Default: 'all'")

parsero = parser.parse_args()

# ================================
# Set INPUTS AND OUTPUTS paths
# ================================
dataset = parsero.dataset                       # Working Directory
input_dir = dataset + "input/"                  # where beads and PICT images are located
pict_images_dir = input_dir + "pict_images/"
beads_dir = input_dir + "out_reg/"
output_dir = dataset + "output/"                # the output of the pipeline will be saved here.
images_dir = output_dir + "images/"             # Pre-processed images are saved here.
spots_dir = output_dir + "spots/"               # Detected spots for each channel and warped coordinates are saved here.
# Output directory to save masks to
segment_dir = output_dir + "segmentations/"     # Segmented yeast masks are saved here.
results_dir = output_dir + "results/"           # Results in csv and txt format are saved here.
figures_dir = output_dir + "figures/"           # All plots are saved here.

# ============
# OPTIONS
# ============
sel_option = parsero.option                     # Part of pipeline to run (default: all)
# PREPROCESSING
bead_registration = False
preprocessing = False
# DETECTION & LINKING
detect_spots = False
mass_selection = False
# AFFINE TRANSFORMATION (WARPING)
warping_transformation = False
global_transformation = True                    # do global affine transformation of channel 1 coordinates
local_transformation = parsero.local_transformation  # do piecewise affine transformation of channel 1 coordinates
if local_transformation:
    global_transformation = False
# SELECTION
segmentation_preprocess = False                 # run yeast cell segmentation
kde = False                                     # run 2D-KDE
gaussian_fit = False                            # run gaussian fitting
mle = False                                     # run MLE and Outlier rejection

# Get options according to user's command line input
if sel_option == 'all':
    bead_registration = True
    preprocessing = True
    detect_spots = True
    warping_transformation = True
    segmentation_preprocess = True
    gaussian_fit = True
    kde = True
    mle = True
if sel_option == 'beads' or 'beads' in sel_option.split(','):
    bead_registration = True
if sel_option == 'pp' or 'pp' in sel_option.split(','):
    preprocessing = True
if sel_option == 'spt' or 'spt' in sel_option.split(','):
    detect_spots = True
if sel_option == 'warping' or 'warping' in sel_option.split(','):
    warping_transformation = True
if sel_option == 'segment' or 'segment' in sel_option.split(','):
    segmentation_preprocess = True
if sel_option == 'kde' or 'kde' in sel_option.split(','):
    kde = True
if sel_option == 'gaussian' or 'gaussian' in sel_option.split(','):
    gaussian_fit = True
if sel_option == 'mle' or 'mle' in sel_option.split(','):
    mle = True

# ==============
# =============================
# START RUNNING PYF2F WORKFLOW
# =============================
# ==============
verbose = parsero.verbose                           # Verbose
dirty = parsero.dirty                               # Save all html plots of the process
px_size = parsero.px_size                           # Pixel size
# IMAGE PRE-PROCESSING
rolling_ball_radius = parsero.rolling_ball_radius
median_filter_radius = parsero.median_filter
# SPOT DETECTION AND LINKING
particle_diameter = parsero.particle_diameter
percentile = parsero.percentile
max_displacement = parsero.max_displacement
# SEGMENTATION
cont_cutoff = parsero.contour_cutoff                # Cutoff to select spots based on distance to contour
neigh_cutoff = parsero.neigh_cutoff                 # Cutoff to select spots based on distance to  closest neighbour
# KDE AND GAUSSIAN
kde_cutoff = parsero.kde_cutoff                     # Select spots in the two channels sharing same properties.
gaussian_cutoff = parsero.gaussian_cutoff           # Spots should fit to a Gaussian-like profile above this cutoff
# OUTLIER REJECTION
mle_cutoff = parsero.mle_cutoff                     # Perform MLE with by bootstrapping the right-tail area = 1 - cutoff
reject_lower = parsero.reject_lower                 # Reject max-scored estimates under this threshold
mu_ini = parsero.mu_ini                             # Initiate the MLE with this initial mu.
sigma_ini = parsero.sigma_ini                       # Initiate the MLE with this initial sigma.
bin_size = parsero.bin_size                         # Bin size for distance distribution plots

# YEAST SPOTTER PARAMS FOR CELL SEGMENTATION
rescale = False                                     # rescale the input images to reduce segmentation time
scale_factor = 2                                    # Factor to downsize images by if rescale is True
save_preprocessed = True                            # save preprocessed images as input to neural network.
save_compressed = False                             # save a compressed RLE version of the masks for sharing
save_masks = False                                  # save the full masks
output_imagej = False                               # output ImageJ-compatible masks
save_contour = True                                 # Save contour images
save_contour_mod = True                             # Save contour modified images

# Set up logging files
if verbose:
    print('\n\n\tRunning PICT-MOD!\n\n'
          '\tYour paths: \n'
          f'\t\t Working directory: {dataset}\n'
          f'\t\t Input directory: {input_dir}\n'
          f'\t\t Output directory: {output_dir}\n\n'
          '\tYour Params: \n'
          f'\t\t Pixel size (nm): {px_size}\n'
          f'\t\t Spot size (px): {particle_diameter}\n'
          f'\t\t Percentile of selected pixels: {percentile}\n'
          f'\t\t Linking 2C (px): {max_displacement}\n'
          f'\t\t Warping: {"global" if global_transformation else "local"}\n'
          f'#########\n'
          )
if not os.path.exists(dataset + "log.txt"):
    logging.basicConfig(filename=dataset + "log.txt", level=logging.DEBUG,
                        format="%(asctime)s %(message)s", filemode="w")
else:
    logging.basicConfig(filename=dataset + "log.txt", level=logging.DEBUG,
                        format="%(asctime)s %(message)s", filemode="a")

logging.info("\n\n############################\n"
             "Image Analysis for LiveCellPICT \n"
             "################################\n\n"
             "\tDataset: {}\n"
             "\tInput directory: {}\n"
             "\tOutput directory: {}\n\n".format(dataset, input_dir, output_dir))

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# Time it!
start = time.time()

# ===================
# 1. PREPROCESSING
# ===================
if preprocessing:
    pp(pict_images_dir, images_dir, rolling_ball_radius,
       median_filter_radius, verbose=verbose)

# ===================
# 2. SPOT DETECTION
# ===================
if detect_spots:
    spot_detection(images_dir, spots_dir, results_dir, figures_dir, particle_diameter, percentile, max_displacement,
                   verbose=verbose, mass_selection=mass_selection, px_size=px_size, dirty=dirty)

# ==========================================================================
# ++ WARPING (AFFINE TRANSFORMATION OF CHANNEL 1 SPOT-CENTROID COORDINATES)
# ==========================================================================
if warping_transformation:
    if verbose:
        print("\n\n#######################\n"
              "Initializing WARPING \n"
              "############################\n\n")
    if local_transformation:
        # Local warping to W1 coordinates of sample images
        local_warping(beads_dir, spots_dir, figures_dir, results_dir, pixel_size=px_size)
    if global_transformation:
        global_warping(beads_dir, spots_dir, figures_dir, results_dir, pixel_size=px_size)
        if dirty:
            plot_links(images_dir + "imageMD*.tif", spots_dir, figures_dir + "spot_detection/")

# =======================================================================
# 3.1 SELECTION: ISOLATION & DISTANCE TO CELL CONTOUR (CELL SEGMENTATION)
# =======================================================================
if segmentation_preprocess:
    total_data, seg_selected = main_segmentation(segment_dir, images_dir, spots_dir, results_dir,
                                                 figures_dir, scale_factor, cont_cutoff,
                                                 neigh_cutoff, rescale=rescale, verbose=verbose,
                                                 px_size=px_size)
    if dirty:
        plot_links(segment_dir + "masks/contour_mod*.tif", spots_dir,
                   figures_dir + "pp_segmented/")

# ====================================================
# 3.2 SELECTION: SHARED BRIGHTNESS PROPERTIES (2D-KDE)
# ====================================================
if kde:
    kde_initial, kde_selected = main_kde(images_dir, results_dir, figures_dir, kde_cutoff,
                                         px_size=px_size, dirty=dirty)
# ====================================================
# 3.3 SELECTION: GOODNESS OF THE GAUSSIAN FIT 
# ====================================================
if gaussian_fit:
    gauss_initial, gauss_selected = main_gaussian(results_dir, images_dir, figures_dir,
                                                  gaussian_cutoff, px_size=px_size, dirty=dirty)
# ====================================================
# 4. OUTLIER REJECTION (MLE) & DISTANCE ESTIMATION
# ====================================================
if mle:
    outlier_rejection(results_dir, figures_dir, images_dir,
                      mu_ini=mu_ini, sigma_ini=sigma_ini, reject_lower=reject_lower,
                      cutoff=mle_cutoff, dirty=dirty, bin_size=bin_size)

# How much time did it take?
end = time.time()
print(f"\n\nTotal Process time: {round(end - start, 3)} s\n")

print("\n\n#######################\n"
      "Done!\n\n"
      "Have a good day :-)\n"
      "#########################\n\n")

sys.exit(0)
# END
