# PyF2F-Ruler

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

**PyF2F-Ruler** is a Python-based software that provides the tools to estimate the distance between two fluorescent markers 
that are labeling one or two protein molecules. This software is the Python implementation of our previous work 
[Picco A., et al, 2017](https://www.sciencedirect.com/science/article/pii/S0092867417300521) where we combined 
[PICT](https://www.sciencedirect.com/science/article/pii/S0092867417300521) (yeast engineering & live-cell imaging)
and integrative modeling to reconstruct the molecular architecture of the exocyst complex in its cellular environment.

How does it work?
-----------

PyF2F-Ruler utilizes bioimage analysis tools that allows for the **pre-processing** (*Background subtraction*, *chromatic
aberration correction*, and *spot detection*) and **analysis** of live-cell imaging (fluorescence microscopy) data. The 
set of image analysis functions can be used to estimate the pair-wise distance between a fluorophores flagging the terminus
of a protein complex (prey-GFP) and a static intracellular anchor site (anchor-RFP). From a dataset of 20 - 30 images, 
PyF2F-Ruler estimates the μ and σ values of the final distance distribution with a precision below 5 nm.


Instructions
-----------
### Use Colab

You can use [PyF2F Colab](https://colab.research.google.com/drive/1kSOnZdwRb4xuznyQIpRNWUBBFKms91M8?usp=sharing)
to run the image analysis workflow without the need of installation. 

### Run it locally

1) Download the git repo in your local computer and get into the folder, or open  terminal and 
   run the following command:

```bash
  $ git clone https://github.com/Altairch95/PICT-MODELLER
  $ cd PyF2F-Ruler
 ```
2) Download the CNN weights: PyF2F-Ruler utilizes the pre-trained weights for the neural network that is used for 
   yeast cell segmentation in [Yeast Spotter](http://yeastspotter.csb.utoronto.ca/). The weights are necessary to run 
   the software, but are too large to share on GitHub. You can download the zip file from this 
   [Zenodo](https://zenodo.org/record/3598690) repository. 

Once downloaded, simply unzip it and move it to the *scripts/* directory. You can also run the following command:

```bash
  $ unzip weights.zip
  $ mv weights/ path/to/PyF2F-Ruler/scripts/
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

At this pont, the directory *PyF2F-Ruler* should contain the files and directories described bellow:

#### Package tree

The package has the following structure:

    PyF2F-Ruler/
      README.md
      scripts/
          calculate_PICT_distances.py
          custom.py
          gaussian_fit.py
          kde.py
          lle.py
          rnd.py
          measure_pict_distances.py  (main script)
          options.py                 (User selections)
          outlier_rejections.py
          segmentation_pp.py
          spot_detection_functions.py
          mrcnn/                      (YeastSpotter)
          weights/                    (for mrcnn yeast segmentation)  
              
      test/
          input/
              pict_images/            (5 images from the exocyst dataset)
              beads/                  (the beads used on the exocyst dataset)


Image Analysis Tutorial
-----------------------

The image analysis can be divided into two steps. First, we processed images to measure the centroid positions of the 
GFP and RFP tags. Then, we analyzed the distribution of these centroid positions to estimate the true separation between
the GFP and RFP fluorophores using Single-molecule High-REsolution Colocalization (SHREC)
([Churchman et al. 2005](https://www.pnas.org/doi/abs/10.1073/pnas.0409487102), 
[Churchman et al. 2006](https://www.sciencedirect.com/science/article/pii/S0006349506722457)).

#### Command line arguments

```bash
  $ python3 measure_pict_distances.py -h
```
        Computing the distance distribution between fluorophores tagging the protein
        complex (e.g, exocyst) with a precision up to 5 nm.
        
        optional arguments:
          -h, --help            show this help message and exit
          -d DATASET, --dataset DATASET
                                Name of the dataset where the input/ directory is
                                located
          --test                Runs the test dataset
          -o OPTION, --option OPTION
                                Option to process: 'all' (whole workflow), 'beads'
                                (bead registration), 'pp' (preprocessing), 'spt' (spot
                                detection and linking), 'warping' (transform XY spot
                                coordinates using the beads warping matrix), 'segment'
                                (yeast segmentation), 'gaussian' (gaussian fitting),
                                'kde' (2D Kernel Density Estimate), 'outlier (outlier
                                rejection using the MLE)'. Default: 'main'


1) Run a test to check that everything is installed and running as expected:

```bash
  $ conda activate {ENV_NAME}  # Make sure to activate your conda environment
  $ python3 measure_pict_distances.py --test 
 ```

The test is composed by 5 raw images located in the *input/pict_images/* folder. By running the test, 
you should visualize on the terminal all the *log* of the image processing and image analysis steps. You 
can also track it on the *log.txt* file.

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

2) Create a directory with the name of your system and the same structure as the *test/*: add to it the containing-beads
   directory (*beads/*) and the containing-PICT-images directory (*pict_images/*).

```bash
  $ mkdir my_dir_name
  $ cd my_dir_name
  # Create beads/ and pict_images/ if not there
  $ mkdir beads/
  $ mkdir pict_images/
  # Move the beads W1.tif and W2.tif to beads/ and your PICT images to pict_images/
  $ mv path/to/beads/*.tif path/to/my_dir_name/beads/
  $ mv path/to/pict-images/*.tif path/to/my_dir_name/pict_images/
 ```

Run the software with your data:

```bash
  $ python3 measure_pict_distances.py -d my_dir 
 ```

You may grab a coffee while waiting for the results :)

Notebooks
--------

Working on it.

Notes
--------

### Note 1: Input files (Beads and PICT images)

This program needs an input of bright-field TIFF images (central quadrant, 16-bit) captured as stacks of two channels: 

  - Channel 1: Red channel    (W1) --> observing RFP spots.
  - Channel 2: Green channel  (W2) --> observing GFP spots. 

**Beads**: To calibrate this protocol, the imaging of [TetraSpeck](https://www.thermofisher.com/order/catalog/product/T7279) 
in both the red and green channel is required. For each channel, the user should acquire images from  
4 fields of view (FOV) with isolated beads (avoiding clusters) and minimizing void areas (each FOV should have 
a homogeneity distribution of beads to cover all the possible coordinates. Finally, the 4 FOV for each channel 
should be stacked (e.g, stack-1 contains frame_FOV_0, frame_FOV_1, frame_FOV_2, frame_FOV_3) and named as **W1** for the
red channel, and **W2** for the green channel.

**PICT images**: *PICT images* is the name used to reference the images gathered from the 
[PICT experiment]((https://www.sciencedirect.com/science/article/pii/S0092867417300521)). Each *pict_image.tif* is a 
stack of red/green channels. Diffraction-limited spots should be visualized when opening the image with ImageJ or any 
other image processing software. 

### Note 2: Running the software

From the input images, the program runs through different steps: 

#### 1) **Bead Registration**:

- *Bead registration*: isolated beads are detected for each channel. Agglomerations of beads, or beads shinning with
   low intensity are excluded based on the 0-th moment <i>M<sub>00</sub></i> of brightness (mass) (excluding the beads 
   with a <i>M<sub>00</sub></i> falling on the *1st* and *95th* percentile).

- *Bead transformation*: selected beads are transformed (aligned) and the transformation matrix is saved. For the 
   alignment, beads selected in W1 (red, mov) are aligned to the beads selected in W2 (green, reference).
        
    > Explanation: because we are imaging the same beads on the red and green channel, the XY coordinates should not
        change. However, because we are imaging at different wavelengths, each channel will refract the light differently
      (the refractive index of the lens varies with wavelength). The inability of the lens to bring the green and red 
       spots into a common focus results in a slightly different image size and focal point for each wavelength. This 
       artifact is commonly known as chromatic aberration, and must be corrected.

#### 2) **Image pre-procesing**:

- *Background subtraction*: Raw PICT images are corrected for the extracellular noise using the Rolling-Ball 
   algorithm. The size for estimating the rolling ball kernel is based on the maximum separation between two yeast 
   cells (a radius around 70 px.)

- *Median filter*: correction for the intracellular noise is also applied with a median filter of 10 px.
   
#### 3) **Spot Detection**:

Diffraction limited spots are detected using [Trackpy](http://soft-matter.github.io/trackpy/). After detection, the spots 
from W1 and W2 channels falling in a maximum range of 1 px are linked (paired). From this step on, each pair will be 
analysed as a couple of red-green spots on the spot selection step.

#### 4) **Spot selection**:

The following steps are meant to refine the dataset and reject as many noisy centroid positions as possible. 

- *Selection of isolated spots close to the cell perimeter*: only isolated pairs close to the anchor sites (located in the 
plasma membrane) are select. Here, yeast cell segmentation is applied to sort spots falling to far from the plasma 
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

The µ, σ and the ensemble of data points that are retained after all these iterations are those that maximize a scoring 
function defined as 

$$
    S (p_{\delta \mu}, p_{\delta \sigma}) = - p_{\mu} \cdot p_{\delta \mu} -  p_{\sigma} \cdot p_{\delta \sigma}
$$

where <i>S(p<sub>δµ</sub>)</i>, <i>S(p<sub>δσ</sub>)</i> will be maimal when both scores <i>p<sub>δµ</sub></i> and 
<i>p<sub>δσ</sub></i> will be similarly maximized.

Acknowledgements
----------


