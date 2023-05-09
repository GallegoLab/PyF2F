"""
============
OPTIONS FILE
============
In this file you will find all the options to run the program PICT-MODELLER,
without the need of using a parser from the terminal. Most options are
already set with a "default value" for "Spot Detection" and "Spot Selection".

Only the name of your dataset (e.g, test) and the option to run (e.g, all) must
be specified by the user.

dataset: Name of the dataset where the input/ directory is located
option: "Option to process:"
         " 'all' (whole workflow),"
         " 'beads' (bead registration),"
         " 'pp' (preprocessing),"
         " 'spt' (spot detection and linking),"
         " 'warping' (transform XY spot coordinates using the beads warping matrix),"
         " 'segment' (yeast segmentation),"
         " 'gaussian' (gaussian fitting),"
         " 'kde' (2D Kernel Density Estimate),"
         " 'outlier (outlier rejection using the MLE)'."
         " Default: 'main'"
"""
import sys
import time
from argparse import ArgumentParser
import matplotlib

matplotlib.use('Agg')

#########
# PARSER
#########

parser = ArgumentParser(
    prog='PICT-MODELLER',
    description='Computing the distance distribution between '
                'fluorophores tagging the protein complex (e.g, exocyst) '
                'with a precision up to 5 nm.')
parser.add_argument("-d", "--dataset",
                    dest="dataset",
                    action="store",
                    type=str,
                    help="Name of the dataset where the input/ directory is located",
                    required=False)

parser.add_argument("--test",
                    dest="test",
                    action="store_true",
                    default=False,
                    help="Runs the test dataset")

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
                         " 'warping' (transform XY spot coordinates using the beads warping matrix),"
                         " 'segment' (yeast segmentation),"
                         " 'kde' (2D Kernel Density Estimate),"
                         " 'gaussian' (gaussian fitting),"
                         " 'outlier (outlier rejection using the MLE)'."
                         " Default: 'main'")

parser.add_argument('--px_size',
                    dest='px_size',
                    action="store",
                    type=int,
                    default=110,
                    help="Pixel size of the camera in nanometers")

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
                    help="Percentile (%) that determines which bright pixels are accepted as spots.")

parser.add_argument('--max_displacement',
                    dest='max_displacement',
                    action="store",
                    type=float,
                    default=1,
                    help="Median Radius (pixels)")

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

parser.add_argument('--mu_ini',
                    dest='mu_ini',
                    action="store",
                    type=int,
                    default=None,
                    help="Initial guess for mu search in the MLE")

parser.add_argument('--sigma_ini',
                    dest='sigma_ini',
                    action="store",
                    type=int,
                    default=None,
                    help="Initial guess for sigma search in the MLE")

parser.add_argument('--reject_lower',
                    dest='reject_lower',
                    action="store",
                    type=float,
                    default=0,
                    help="In the MLE, reject selected values under this threshold")

parser.add_argument('--mle_cutoff',
                    dest='mle_cutoff',
                    action="store",
                    type=float,
                    default=2/3,
                    help="In the MLE, % of the distribution assumed to be ok. "
                         "Outlier search in the right 1 - value area of the distance distribution")

parser.add_argument('--beads_registration',
                    dest='test_beads_registration',
                    action="store_true",
                    help="Set this option to test beads registration")

parser.add_argument("--rfp_channel",
                    dest="rfp_channel",
                    action="store",
                    type=int,
                    default=0,
                    help="Define the RFP channel index")

parser.add_argument("--gfp_channel",
                    dest="gfp_channel",
                    action="store",
                    type=int,
                    default=1,
                    help="Define the GFP channel index")

parsero = parser.parse_args()

# ===================
# INPUTS AND OUTPUTS
# ====================
if parsero.test:
    dataset = "test"
    sel_option = "all"
elif parsero.dataset is None:
    sys.stderr.write("\n\n\tPICT-MODELLER-WARNING: Please, specify the name of your directory. Thanks!\n\n")
    time.sleep(2)
    parser.print_help()
    sys.exit(1)

elif parsero.test_beads_registration:
    dataset = parsero.dataset
    sel_option = "beads,spt,warping,outlier"
else:
    dataset = parsero.dataset
    sel_option = parsero.option

# Working Directory
working_dir = f"{dataset}"
input_dir = working_dir + "input/"
output_dir = working_dir + "output/"

# Spot Detection Paths
# Input directory of images to be segmented
pict_images_dir = input_dir + "pict_images/"  # *** paste your PICT images here
beads_dir = input_dir + "reg_out/"  # *** paste your bead stacks here
images_dir = output_dir + "images/"
warped_dir = output_dir + "warped/"
spots_dir = output_dir + "spots/"
# Output directory to save masks to
segment_dir = output_dir + "segmentations/"
figures_dir = output_dir + "figures/"
results_dir = output_dir + "results/"

# ======================

###########
# OPTIONS
# #########
# PREPROCESSING: Background Subtraction, Medial Filter, Warping
bead_registration = False
preprocessing = False
# DETECTION & LINKING
detect_spots = False
mass_selection = False
# WARPING
warping = False
# SELECTION: Segmentation, Gaussian Fitting, KDE, Outlier Rejection
segmentation_preprocess = False
gaussian_fit = False
kde = False
outlier_rejection = False

if sel_option == 'all':
    bead_registration = True
    preprocessing = True
    detect_spots = True
    warping = True
    segmentation_preprocess = True
    gaussian_fit = True
    kde = True
    outlier_rejection = True
if sel_option == 'beads' or 'beads' in sel_option.split(','):
    bead_registration = True
if sel_option == 'pp' or 'pp' in sel_option.split(','):
    preprocessing = True
if sel_option == 'spt' or 'spt' in sel_option.split(','):
    detect_spots = True
if sel_option == 'warping' or 'warping' in sel_option.split(','):
    warping = True
if sel_option == 'segment' or 'segment' in sel_option.split(','):
    segmentation_preprocess = True
if sel_option == 'kde' or 'kde' in sel_option.split(','):
    kde = True
if sel_option == 'gaussian' or 'gaussian' in sel_option.split(','):
    gaussian_fit = True
if sel_option == 'outlier' or 'outlier' in sel_option.split(','):
    outlier_rejection = True


# ======================

###########
# PARAMETERS
# #########
# Pixel size of the camera
px_size = parsero.px_size
# TEST BEADS REGISTRATION
test_beads_registration = parsero.test_beads_registration
# IMAGE PRE-PROCESSING: Background Subtraction, Medial Filter, Warping
rfp_ch = parsero.rfp_channel
gpf_ch = parsero.gfp_channel
rolling_ball_radius = parsero.rolling_ball_radius
median_filter_radius = parsero.median_filter
# SPOT DETECTION AND LINKING
particle_diameter = parsero.particle_diameter  # must be an odd number.
percentile = parsero.percentile  # float. Percentile (%) that determines which bright pixels are accepted as spots.
max_displacement = parsero.max_displacement # LINK PARTICLES INTO TRAJECTORIES
min_mass_cutoff = 0.01
max_mass_cutoff = 0.95

# SEGMENTATION:
# Cutoffs to select spots based on distance to contour and closest neighbour
cont_cutoff = parsero.contour_cutoff
neigh_cutoff = parsero.neigh_cutoff

# KDE AND GAUSSIAN
kde_cutoff = parsero.kde_cutoff
gaussian_cutoff = parsero.gaussian_cutoff

# OUTLIER REJECTION
# Deprecated, know using the median and std from the raw distribution of distances
# to start the optimization
mu_ini = parsero.mu_ini
sigma_ini = parsero.sigma_ini
mle_cutoff = parsero.mle_cutoff
reject_lower = parsero.reject_lower

# Set to true to rescale the input images to reduce segmentation time
rescale = False
scale_factor = 2  # Factor to downsize images by if rescale is True

# Set to true to save preprocessed images as input to neural network (useful for debugging)
save_preprocessed = True

# Set to true to save a compressed RLE version of the masks for sharing
save_compressed = False

# Set to true to save the full masks
save_masks = True

# Set to true to have the neural network print out its segmentation progress as it proceeds
verbose = True

# Set to true to output ImageJ-compatible masks
output_imagej = False

# Save contour images
save_contour = True

# Save contour modified images
save_contour_mod = True
