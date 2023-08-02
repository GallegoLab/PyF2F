#!/usr/bin/env python
# coding: utf-8

import os
import glob
import shutil
import sys
import time
import pandas as pd
import numpy as np
import trackpy as tp
import pims
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import tkinter as tk
from tkinter import filedialog
from skimage import io, util
from skimage import data
from scipy import stats
from scipy import spatial
from pymicro.view.vol_utils import compute_affine_transform
from matplotlib import rcParams
import argparse


from detect_beads import *

#########
# PARSER
#########
p = argparse.ArgumentParser(
    description="Run bead registration"
)
p.add_argument('-bri', action="store", dest="path_beads_reg",
               type=str,
               default=None,
               required=True,
               help="Path to beads registration set")

p.add_argument('-bti', action="store", dest="path_beads_test",
               type=str,
               default=None,
               required=True,
               help="Path to beads test set")

p.add_argument('-bro', action="store", dest="path_output_reg",
               type=str,
               default="./output_reg/",
               required=False,
               help="Path to output coordinates for bead registration set")

p.add_argument('-bto', action="store", dest="path_output_test",
               type=str,
               default="./output_test/",
               required=False,
               help="Path to output coordinates for bead test set")

p.add_argument('--name', action="store", dest="beads_head",
               type=str,
               default="*.tif",
               required=False,
               help="Pattern in bead images names used to load images. E.g, Beads*. It is case sensitive.")

p.add_argument('-px', action="store", dest="px_size",
               default="110",
               required=False,
               help="Pixel size in nm of the camera")

p.add_argument('-dbpsf', action="store", dest="db_psf",
               type=int,
               default="700",
               required=False,
               help="Double of the PSF MHHW in nm to calculate spot size")

p.add_argument('-sd', action="store", dest="spot_diameter",
               type=int,
               default=None,
               required=False,
               help="Spot diameter in nm for spot detection. If not specified, spot spot_diameter"
               " is calculated as sd = dbPSF / px_size")

p.add_argument('-pc', action="store", dest="percentile",
               type=float,
               default=99.8,
               required=False,
               help="Percentile. Spots which intensity falls above this pc are selected")


p.add_argument('--global', action="store_true", dest="global_registration",
               required=False,
               help="Use it to apply global registration. If used, the parameters used for "
               "the local affine transformation are not used.")

p.add_argument('--min_mass', action="store", dest="min_mass",
               type=float,
               default=0.01, 
               required=False,
               help="Min mass to select spots. See Trackpy's documentation")

p.add_argument('--max_mass', action="store", dest="max_mass",
               type=float,
               default=0.95,
               required=False,
               help="Max mass to select spots. See Trackpy's documentation")

p.add_argument('-l', action="store", dest="max_displacement",
               type=int,
               default=1,
               required=False,
               help="Max separation in px to link spots between the two channels")


p.add_argument('--min_fiducials', action="store", dest="min_fiducials",
               type=int,
               default=10,
               required=False,
               help="Min number of fiducials for piecewise affine correction. Defualt is 10.")

p.add_argument('-s', action="store", dest="search_range",
               type=int,
               default=2000,
               required=False,
               help="Search range in nm for piecewise affine correction.")

p.add_argument('--plot', action="store_true", dest="plot",
               required=False,
               help="Use it to plot histograms for each field of view with detect_beads and link beads.")


parsero = p.parse_args()

#############
# PATHS
#############
path_beads_reg = parsero.path_beads_reg
path_beads_test = parsero.path_beads_test
if path_beads_reg is None or path_beads_test is None:
    sys.stderr.write("Please specify the directoris to create the registration map (-bri setA) and to test the registration (-bti setB):\n")
    sys.exit(1)

# Create directories for output
path_output_reg = parsero.path_output_reg
path_output_test = parsero.path_output_test

if not os.path.exists(path_output_reg):
    os.mkdir(path_output_reg)
    
if not os.path.exists(path_output_test):
    os.mkdir(path_output_test)


print("\n\tYour Paths: \n"
          "\t\t -Beads_Reg directory: {}\n"
          "\t\t -Beads_Test directory: {}\n"
          "\t\t -Output_Reg directory: {}\n"
          "\t\t -Output_Test directory: {}\n".format(path_beads_reg,
                                                 path_beads_test,
                                                 path_output_reg,
                                                 path_output_test))
#############
# PARAMETERS
#############
beads_head = parsero.beads_head                         # beads head for loading bead images
spot_diameter = parsero.spot_diameter                   # spot diameter
px_size = float(parsero.px_size)                        # px size in nm
print(f"\n\tPyF2F message: Pixel size is {px_size}\n")
if spot_diameter is None:
    if px_size == 64.5:                                 # spot detection: diameter of spots in px
        spot_diameter = 11                              # Spot size Zyla
    elif px_size == 110:
        spot_diameter = 7                               # Spot size Prime

print(f"\n\tPyF2F message: Spot diameter is {spot_diameter}\n")


percentile = float(parsero.percentile)                # spot detection: sort spots below this percentile of intensity 
min_mass = float(parsero.min_mass)                    # spot detection: sort spots with a mass above this threshold (range 0-1). 
max_mass = float(parsero.max_mass)                    # spot detection: sort spots with a mass below this threshold (range 0-1).
max_displacement=int(parsero.max_displacement)        # linking: link spots from ch1-ch2 channels separated by this cutoff in px
search_range = int(parsero.search_range) // px_size   # local registration: max distance in px for nearest-neighbour search
min_fiducials = int(parsero.min_fiducials)            # local registration: minimum number of fiducial markers to correct locally 
do_plots = parsero.plot                                                 


# Now, we can calculate the Target Registration Error (TRE).

# ## Calculate TRE
# Get two channels (c1-c2) coordinates from the two sets of beads to a) calculate the registration map (REF) and b) to calculate the TRE (TEST).
# If the coordinates have been previously calculated, then are loaded from the saved CSV files in the "output_reg" and "output_test" directories.

######################
# 1. Get Coordinates
######################
# Check if spots have been previously detected and saved in 
# the "path_output_reg" and "path_output_test" directories  
if os.path.exists(path_output_reg + "/coords_W1.csv") and os.path.exists(path_output_reg + "/coords_W2.csv") and not len([c for c in open(path_output_reg + "/coords_W1.csv")]) > 10 and not len([c for c in open(path_output_reg + "/coords_W2.csv")]) > 10:
    os.remove(path_output_reg + "/coords_W1.csv")
    os.remove(path_output_reg + "/coords_W2.csv")

if os.path.exists(path_output_test + "/coords_W1.csv") and os.path.exists(path_output_test + "/coords_W2.csv") and not len([c for c in open(path_output_test + "/coords_W1.csv")]) > 10 and not len([c for c in open(path_output_test + "/coords_W2.csv")]) > 10:
    os.remove(path_output_test + "/coords_W1.csv")
    os.remove(path_output_test + "/coords_W2.csv")

if os.path.exists(path_output_reg + "/coords_W1.csv") and os.path.exists(path_output_reg + "/coords_W2.csv"):
    c1_ref = np.loadtxt(path_output_reg + "/coords_W1.csv", delimiter=",")
    c2_ref = np.loadtxt(path_output_reg + "/coords_W2.csv", delimiter=",")
else:
    # Get coordinates from beads for registration
    c2_ref, c1_ref = get_coords(path_output_reg, path_beads_reg, beads_head, diameter=spot_diameter, separation=max_displacement, percentile=percentile,
                                min_mass_cutoff=min_mass, max_mass_cutoff=max_mass, px_size=px_size, plot=do_plots)

if os.path.exists(path_output_test + "/coords_W1.csv") and os.path.exists(path_output_test + "/coords_W2.csv"):
    c1_test = np.loadtxt(path_output_test + "/coords_W1.csv", delimiter=",")
    c2_test = np.loadtxt(path_output_test + "/coords_W2.csv", delimiter=",")
    
else:
    # Get coordinates from beads for registration
    c2_test, c1_test = get_coords(path_output_test, path_beads_test, beads_head, diameter=spot_diameter, separation=max_displacement, percentile=percentile,
                                  min_mass_cutoff=min_mass, max_mass_cutoff=max_mass, px_size=px_size, plot=do_plots)

plotly_coords(c1_test, c2_test, path=path_output_reg, title=f"TEST\n N={len(c1_test)}")


###########################
# 2. Affine Transformation
###########################
global_affine = parsero.global_registration
c1_test_new = np.empty_like(c1_test)
if global_affine:
    #####################
    # Global Affine
    ######################
    print("\nUsing Global Affine Transformation\n")
    # compute the global affine transform from the point set
    translation, transformation = compute_affine_transform(c2_ref, c1_ref)
    ref_centroid = np.mean(c2_ref, axis=0)
    mov_centroid = np.mean(c1_ref, axis=0)
    # Save transformation matrices for later affine correction
    np.save(path_output_reg + '/transform.npy', transformation)
    # correct bead test dataset 
    for i in range(len(c1_test)):
        c1_test_new[i] = ref_centroid + np.dot(transformation, c1_test[i] - mov_centroid)

else:
    ######################
    # Piecewise Affine
    ######################
    print("\nUsing Piecewise Affine Transformation\n")
    # Apply local affine (piecewise) transformation to the TEST dataset from the REF dataset

    print("  - Your Search Range Parameters for Local Affine are:\n"
         f"  \t Search Range: {search_range} nm\n"
         f"  \t Minim Fiducials: {min_fiducials}\n"
         f"  \t Pixel Size is: {px_size} nm")

    for i in range(len(c1_test)):
        c1_test_new[i] = piecewise_affine(c1_test[i], c1_ref, c2_ref, search_range=search_range,
                                          min_candidates=min_fiducials)
        if c1_test_new[i][0] == np.nan:
            sys.stderr.write(f"Coord {c1_test[i]} cannot be locally corrected, no neighbours found!\n")

# Plot and save coordinates
plotly_coords(c1_test, c2_test, path=path_output_test, c1_corrected=c1_test_new, title=f"TEST\n N={len(c1_test)}")



# Calculate the Target Registration Error and plot the distances between c1-c2 TEST beads before and after the piecewise affine transformation

######################
# 3. Calculate TRE
######################
plot_distance_distribution(path_output_test, c2_test, c1_test, c1_test_new, px_size=px_size)
calculate_tre(path_output_test, c2_test, c1_test, c1_test_new, px_size=px_size)

print("\n\nDone!\n\n")
sys.exit(0)

#END
