<h1 align="center">PyF2F</h1>
<h3 align="center">Robust and simplified fluorophore-to-fluorophore distance measurements</h3>
<p align="center">
  <a href="/LICENSE" alt="licence"><img src="https://img.shields.io/github/license/GallegoLab/PyF2F"></a>
  <a href="https://zenodo.org/badge/latestdoi/638469280" alt="DOI"><img src="https://zenodo.org/badge/638469280.svg"></a>
  <a href="https://www.python.org/downloads/release/python-370/"><img src="https://img.shields.io/badge/python-3.7-blue" alt="Python version"></a>
  <a href="https://colab.research.google.com/drive/1kSOnZdwRb4xuznyQIpRNWUBBFKms91M8?authuser=2&pli=1"> <img src="https://colab.research.google.com/assets/colab-badge.svg"> PyF2F Colab </a>
</p>



Contents
-----------

<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->
- [What is it?](#what-is-it)
- [How does it work?](#how-does-it-work)
- [Instructions](#instructions)
  - [Use Colab](#use-colab)
  - [Run it locally](#run-it-locally)
- [Notebooks](#notebooks)

<!-- /TOC -->

What is it?
-----------

**PyF2F** is a Python-based software that provides the tools to estimate the distance between two fluorescent markers 
that are labelling the termini of a protein molecule and a static anchoring platform in the cell. This software is the 
Python implementation of our previous work [Picco A., et al, 2017](https://www.sciencedirect.com/science/article/pii/S0092867417300521) 
where we combined [PICT](https://www.sciencedirect.com/science/article/pii/S0092867417300521) (yeast engineering & live-cell imaging)
and integrative modelling to reconstruct the molecular architecture of the exocyst complex in its cellular environment.

How does it work?
-----------

PyF2F utilises bioimage analysis tools that allows for the **pre-processing** (*Background subtraction*, *chromatic
aberration correction*, and *spot detection*) and **analysis** of live-cell imaging (fluorescence microscopy) data. The 
set of image analysis functions can be used to estimate the pairwise distance between a fluorophores flagging the terminus
of a protein complex (prey-GFP) and a static intracellular anchor site (anchor-RFP). From a dataset of 20 - 30 images, 
PyF2F estimates the μ and σ values of the final distance distribution with a precision below 5 nm.


Instructions
-----------
### <a href="https://colab.research.google.com/drive/1kSOnZdwRb4xuznyQIpRNWUBBFKms91M8?authuser=2&pli=1"> <img src="https://colab.research.google.com/assets/colab-badge.svg"> PyF2F Colab </a>

You can use the Colab to run the image analysis workflow online. 

### Run it locally

#### Requirements
  * Python version 3.7
  * All requirements list in `requirements.txt`


1) Download the git repo in your local computer and get into the folder, or open  terminal and 
   run the following command:

```bash
  $ git clone https://github.com/GallegoLab/PyF2F.git
  $ cd PyF2F
 ```
2) Download the CNN weights: PyF2F utilises the pre-trained weights for the neural network that is used for 
   yeast cell segmentation in [Yeast Spotter](http://yeastspotter.csb.utoronto.ca/). The weights are necessary to run 
   the software, but are too large to share on GitHub. You can download the zip file from this 
   [Zenodo](https://zenodo.org/record/3598690) repository.

   You can also download the CNN weights for the yeast cell segmentation with the `zenodo-get` command:

```bash
  $ pip install zenodo-get
  $ zenodo_get -r 3598690
 ```

  Once downloaded, simply unzip it and move it to the *scripts/* directory. You can also run the following command:

```bash
  $ unzip weights.zip
  $ rm weights.zip
  $ mv weights/ scripts/
 ```

3) Create a conda environment with Python3.7:

```bash
  $ conda create -n {ENV_NAME} python=3.7 anaconda
  $ conda activate {ENV_NAME}
 ```

4) Install the requirements listed in *requirements.txt*:

```bash
  $ pip install -r requirements.txt
 ```

At this point, the directory *PyF2F* should contain the files and directories described below:

#### PyF2F tree

PyF2F has the following structure:

    PyF2F/
      README.md
      scripts/
          run_pyf2f.sh               (running PyF2F using a bash script)
          functions.py               
          custom.py
          gaussian_fit.py
          kde.py
          lle.py
          rnd.py
          Pyf2f_main.py              (main script)
          options.py                 (User selections)
          outlier_rejections.py
          segmentation_pp.py
          spot_detection_functions.py
          mrcnn/                      (YeastSpotter)
          weights/                    (for mrcnn yeast segmentation)  

      scripts_colab/                  (Scripts adapted to run in Colab)
              
      full_example/
          input/
              pict_images/            (21 images from Picco et al., 2017)
              reg/                    (beads set to create the registration map)
              test/                   (beads set to test the registration)

      short_example/
          input/
              pict_images/            (4 images from the full_example dataset)
              reg/                    (beads set to create the registration map)
              test/                   (beads set to test the registration)


Image Analysis Tutorial
-----------------------

The image analysis can be divided into two steps. First, we processed images to measure the centroid positions of the 
GFP and RFP tags. Then, we analysed the distribution of these centroid positions to estimate the true separation between
the GFP and RFP fluorophores using Single-molecule High-REsolution Colocalization (SHREC)
([Churchman et al. 2005](https://www.pnas.org/doi/abs/10.1073/pnas.0409487102), 
[Churchman et al. 2006](https://www.sciencedirect.com/science/article/pii/S0006349506722457)).

#### Command line arguments

```bash
  $ python3 Pyf2f_main.py -h

Pipeline to estimate the distance between a fluorophore fused to the termini
of a protein complex and a reference fluorophore in the anchor in living
yeast(see PICT method in Picco et al., 2017)

optional arguments:
  -h, --help            show this help message and exit
  -d DATASET, --dataset DATASET
                        Name of the main directory where the dataset is
                        located
  --px_size PX_SIZE     Pixel size of the camera (nanometers)
  --rolling_ball_radius ROLLING_BALL_RADIUS
                        Rolling Ball Radius (pixels)
  --median_filter MEDIAN_FILTER
                        Median Radius (pixels)
  --particle_diameter PARTICLE_DIAMETER
                        For spot detection. Must be an odd number
  --percentile PERCENTILE
                        Percentile that determines which bright pixels are
                        accepted as spots.
  --max_displacement MAX_DISPLACEMENT
                        Median Radius (pixels)
  --local_transformation
                        Use local affine (piecewise) transformation instead of
                        global affine.
  --contour_cutoff CONTOUR_CUTOFF
                        Max distance to cell contour (pixels)
  --neigh_cutoff NEIGH_CUTOFF
                        Max distance to closest neighbour (pixels)
  --kde_cutoff KDE_CUTOFF
                        Spots with this probability to be found in the
                        population
  --gaussian_cutoff GAUSSIAN_CUTOFF
                        Spots with this probability to be found in the
                        population
  --mle_cutoff MLE_CUTOFF
                        In the MLE, percentage of the distribution assumed
                        to be ok. Outlier search in the right '1 - value' area
                        of the distance distribution
  --reject_lower REJECT_LOWER
                        In the MLE, reject selected values under this
                        threshold
  --mu_ini MU_INI       Initial guess for mu search in the MLE
  --sigma_ini SIGMA_INI
                        Initial guess for sigma search in the MLE
  --dirty               Generates an html file to show the spots
                        selected/rejected in each image for each step of the
                        process. Consumes more time, memory, and local space.
  --verbose             Informs in the terminal what the program is doing at
                        each step
  -o OPTION, --option OPTION
                        Option to process: 'all' (whole workflow), 'beads'
                        (bead registration), 'pp' (preprocessing), 'spt' (spot
                        detection and linking), 'warping' (transform XY spot
                        coordinates using the beads registration map),
                        'segment' (yeast segmentation), 'kde' (2D Kernel
                        Density Estimate), 'gaussian' (gaussian fitting), 'mle
                        (outlier rejection using the MLE)'. Default: 'all'
```


Run PyF2F-Ruler with the `short_example` dataset to check that everything works properly:

  a) Image Registration Workflow (creating the Registration map)

```bash
  $ cd scripts                                                                         # make sure you are in scripts/ folder
  $ conda activate {ENV_NAME}                                                          # make sure to activate your conda environment
  $ bash run_image_registration.sh ../short_example/input/ reg out out_reg out_test    # create registration map
```
  The resulting transformation matrix and registration map will be saved in 
  the `out_reg` and `out_test` folders, respectively.

  b) Distance Estimation Workflow

```bash
  $ bash run_pyf2f.sh ../short_example/ all                                             # Run all the workflow steps to estimate the distance
                                                                                        # between fluorophores.  
```

* The `short_example` is composed by 4 PICT images located in the *short_example/input/pict_images/* folder.
* The `full_example` is composed by 21 PICT images located in the *full_example/input/pict_images/* folder. 

The results are generated and saved in the *output/* folder with different sub-folders:
<ul>
    <li>images: contains the processed images.</li>
    <li>spots: contains the data from spot detection on your PICT images.</li>
    <li>segmentation: contains the segmented images, masks, and contour images.</li>
    <li>results: contains the resulting files from each processing/analysis step</li>
    <li>figures: contains HTML and png files to get track of the detected and selected spots for each 
    image, on each step, as well as the distance distribution for each step. It also contains PDF file with
    the final distance distribution and params estimates (mu and sigma) </li>
</ul>

You may grab a coffee while waiting for the results :)

Tutorials
--------

You may be interested in running the PyF2F workflow directly in Colab. By default, Colab 
runs through the `small_dataset`

In the `notebooks` directory you will the jupyter-notebooks to run the tutorias for:

- Image Registration: PyF2F_image_registration.ipynb

* A full explanation of the workflow can be found in the notebook *Registration_tutorial_with_test.ipynb*

- Distance Estimation: PyF2F_Estimate_Distances_Walkthrough.ipynb 

You can also run the whole workflow using the Colab notebook called *PyF2F_Ruler_Colab.ipynb*


Reporting Bugs 
---------------
Any bug or error that may appear while running PyF2F, please contact altair.chinchilla@upf.edu

Copyright
-----------
This software includes the Boostrap Copyright(C) 2023 Andrea Picco and the Mask R-CNN Copyright (C) 2017 Matterport, Inc.

  > Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

