<h1 align="center">PyF2F</h1>
<h3 align="center">Robust and simplified fluorophore-to-fluorophore distance measurements</h3>
<p align="center">
  <a href="/LICENSE" alt="licence"><img src="https://img.shields.io/github/license/GallegoLab/PyF2F"></a>
  <a href="https://zenodo.org/badge/latestdoi/638469280" alt="DOI"><img src="https://zenodo.org/badge/638469280.svg"></a>
  <a href="https://www.python.org/downloads/release/python-370/"><img src="https://img.shields.io/badge/python-3.7-blue" alt="Python version"></a>
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
- [Notes](#notes)
  - [Note 1: Input files](#note-1-input-files-beads-and-pict-images)
  - [Note 2: Running the Software](#note-2-running-the-software)
    - [Bead registration](#1-bead-registration)
    - [Pre-Processing](#2-image-pre-procesing)
    - [Spot Detection](#3-spot-detection)
    - [Spot Selection](#4-spot-selection)

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
### <a href="https://colab.research.google.com/drive/1kSOnZdwRb4xuznyQIpRNWUBBFKms91M8?usp=sharing"> <img src="https://colab.research.google.com/assets/colab-badge.svg"> PyF2F Colab </a>

You can use the Colab to run the image analysis workflow online. 

### Run it locally

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

3) Create a conda environment:

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
              
      example/
          input/
              pict_images/            (21 images from Picco et al., 2017)
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


1) Run PyF2F-Ruler with the `example` dataset:

```bash
  $ conda activate {ENV_NAME}  # Make sure to activate your conda environment
  $ python3 Pyf2f_main.py [PARAMS][OPTIONS]
 ```

The `example` is composed by 21 raw images located in the *input/pict_images/* folder. 

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

2) Create a directory with the name of your system and the same structure as the *example/*: add to it the beads
   images to create the registration map (*reg/*) and another beads set to test the registration error (*reg/*).
   Put your images in the directory called *pict_images/*.

```bash
  $ mkdir my_dir_name
  $ cd my_dir_name
  # Create reg/ and pict_images/ if not there
  $ mkdir reg/
  $ mkdir test/
  $ mkdir pict_images/
  # Move the beads W1.tif and W2.tif to beads/ and your PICT images to pict_images/
  $ mv path/to/beads-set-1/*.tif path/to/my_dir_name/reg/
  $ mv path/to/beads-set-2/*.tif path/to/my_dir_name/test/
  $ mv path/to/pict-images/*.tif path/to/my_dir_name/pict_images/
 ```

Run the software with your data using the bash script `run_pyf2f.sh `:

```bash
  $ bash run_pyf2f.sh my_dir 
 ```
** Check carefully the parameters in the bash script before running it.

You may grab a coffee while waiting for the results :)

Tutorials
--------

In the `notebooks` directory you will the jupyter-notebooks to run the tutorias for:

- Image Registration: PyF2F_image_registration.ipynb

* A full explanation of the workflow can be found in the notebook *Registration_tutorial_with_test.ipynb*

- Distance Estimation: PyF2F_Estimate_Distances_Walkthrough.ipynb 

You can also run the whole workflow using the Colab notebook called *PyF2F_Ruler_Colab.ipynb*

Notes
--------

### Note 1: Input files (Beads and PICT images)

This program needs an input of bright-field TIFF images (central quadrant, 16-bit) captured as stacks of two channels: 

  - Channel 1: Red channel    (W1) --> observing RFP spots.
  - Channel 2: Green channel  (W2) --> observing GFP spots. 

**Beads**: To calibrate this protocol, the imaging of [TetraSpeck](https://www.thermofisher.com/order/catalog/product/T7279) 
in both the red and green channel is required. For each channel, the user should acquire images from  
*n* different fields of view (FOV) with isolated beads (avoiding clusters) and minimising void areas (each FOV should have 
a homogeneity distribution of beads to cover all the possible coordinates. Finally, the FOV for each channel 
should be stacked (e.g, stack-1 contains frame_FOV_0, frame_FOV_1, frame_FOV_2, frame_FOV_3).

**PICT images**: *PICT images* is the name used to reference the images gathered from the 
[PICT experiment]((https://www.sciencedirect.com/science/article/pii/S0092867417300521)). Each *pict_image.tif* is a 
stack of red/green channels. Diffraction-limited spots should be visualised when opening the image with ImageJ or any 
other image processing software. 

### Note 2: Running the software

From the input images, the program runs through different steps: 

#### 1) **Bead Registration**:

- *Bead registration*: isolated beads are detected for each channel. Agglomerations of beads, or beads shining with
   low intensity are excluded based on the 0-th moment <i>M<sub>00</sub></i> of brightness (mass) (excluding the beads 
   with a <i>M<sub>00</sub></i> falling on the *1st* and *95th* percentile).

- *Bead transformation*: selected beads are transformed (aligned) and the transformation matrix is saved. For the 
   alignment, beads selected in W1 (red, mov) are aligned to the beads selected in W2 (green, reference).
        
    > Explanation: because we are imaging the same beads on the red and green channel, the XY coordinates should not
        change. However, because we are imaging at different wavelengths, each channel will refract the light differently
      (the refractive index of the lens varies with wavelength). The inability of the lens to bring the green and red 
       spots into a common focus results in a slightly different image size and focal point for each wavelength. This 
       the artifact is commonly known as chromatic aberration, and must be corrected.

#### 2) **Image pre-procesing**:

- *Background subtraction*: Raw PICT images are corrected for the extracellular noise using the Rolling-Ball 
   algorithm.

- *Median filter*: correction for the uneven illumination coming from the intracellular noise is also applied by 
   subtracting the median-filtered image to the background-subtracted image. 
   
#### 3) **Spot Detection**:

Diffraction limited spots are detected using [Trackpy](http://soft-matter.github.io/trackpy/). After detection, the spots 
from W1 and W2 channels falling in a maximum range of *x* px are linked (paired). From this step on, each pair will be 
analysed as a couple of red-green spots on the spot selection step.

#### 4) **Spot selection**:

The following steps are meant to refine the dataset and reject as many noisy centroid positions as possible. 

- *Selection of isolated spots close to the cell perimeter*: only isolated pairs close to the anchor sites (located in the 
plasma membrane) are selected. Here, yeast cell segmentation is applied to sort spots falling to far from the plasma 
membrane, close to the neck of yeast cells, or too close to its closest neighbour.

- *Selection of the spots in focus*: detected pairs (couples) of spots are analysed according to the second moment <i>m<sub>2</sub></i> of 
brightness (size) and eccentricity. We assume that the majority of the spots are in focus. The spots in focus will thus 
cluster in a two-dimensional space defined by the second moments of brightness and the eccentricity for each GFP and RFP 
spot pairs. The clustering spots are identified with a 2D binned kernel density estimate (KDE) setting a threshold of 50% 
or higher of total probability to select the most likely pair of spots to be found in the population. The spots selected 
are those clustering in both the GFP and the RFP channels.

- *Refinement of the spot selection*: A 2D gaussian approximates the point spread function which describes the distribution 
of the pixel fluorescence intensity in each spot. A 2D gaussian must thus fit well both GFP and RFP spots. We performed 
a goodness of gaussian fit on each GFP and RFP spot pairs and retained only the spot pairs with an 
<i>R<sup>2</sup></i> > 0.35

- *MLE and outlier rejection*: The distribution of distance measurements should approximate a known non-Gaussian 
distribution described in [Churchman et al. 2006](https://www.sciencedirect.com/science/article/pii/S0006349506722457)):

$$  
\begin{align}
    p (d) = \left ( \cfrac{d}{2 \pi \sigma^2} \right ) \textup{exp} -\left ( \cfrac{\mu^2 + d^2}{2 \sigma^2} \right ) 
    \ I_0 \left ( \cfrac{d \ \cdot\mu }{\sigma^2} \right)
\end{align}
$$

where <i>I<sub>0</sub></i> is the Bessel function of order 0. The true separation between the fluorophores can thus be 
computed with a MLE. Because of the skewed nature of the distribution, outliers, especially if in the tail of the 
distribution, can fail the MLE. To reject those, we proceeded with a bootstrap method. Each datapoint is rejected, 
one at the time, and for each rejection we compute the log-likelihood given an initial estimate of µ and σ. The datapoint 
that is less likely to belong to the dataset is the one whose rejection gives the worst log-likelihood. This datapoint is 
rejected and a new estimate of µ and σ is computed. The process is iterated until one third of the dataset, starting from 
the largest distances (which are those defining the tail of the distribution, where outliers, if present, are more 
problematic), has been sampled for rejection (we do not expect to reject as many data points, but 1/3 seemed a safe 
parameter to ensure that we had a large sampling of the dataset). Two subsequent rejections <i>i</i> and <i>i</i> + 1 
will give two estimates of µ. Their difference, δµ<sub>i</sub> = µ<sub>i + 1</sub> - µ<sub>i</sub> will decrease when 
most outliers are rejected and the score:

$$
    p_{\delta \mu} = \cfrac{\cfrac{1}{\delta \mu}}{\sum \delta \mu} \ 
    \textup{log} \left ( \cfrac{\cfrac{1}{\delta \mu}}{\sum \delta \mu} \right )
$$

will thus be maximal.

The µ, σ and the ensemble of data points that are retained after all these iterations are those that maximise a scoring 
function defined as 

$$
    S (p_{\delta \mu}, p_{\delta \sigma}) = - p_{\mu} \cdot p_{\delta \mu} -  p_{\sigma} \cdot p_{\delta \sigma}
$$

where <i>S(p<sub>δµ</sub>)</i>, <i>S(p<sub>δσ</sub>)</i> will be maximal when both scores <i>p<sub>δµ</sub></i> and 
<i>p<sub>δσ</sub></i> will be similarly maximised.

Copyright
-----------
This software includes the Boostrap Copyright(C) 2023 Andrea Picco and the Mask R-CNN Copyright (C) 2017 Matterport, Inc.

  > Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

