#!/usr/bin/bash
# Description: This is a bash script to run the image registration
# protocol of PyF2F. The parameters below are adequated for a camera 
# of pixel size = 64.5 nm

# PARAMETERS 
directory=$1               # name of your sample directory
px_size=64.5               # pixel size in nm (Zyla: 65; Prime: 110)
rb_radius=70               # Rolling ball radius
md_radius=10		   # Median filter radius
diameter=11	           # spot detection: diameter of spots in px
percentile=99.7		   # spot detection: sort spots below this percentile of intensity (select only the brightest spots)
max_displacement=2         # linking: link spots from W1-W2 channels separated by a 'max_displacement' in px
contour_cutoff=13	   # segmentation: select spots falling no far from this cutoff to the cell contour
neigh_cutoff=9	  	   # segmentation: select spots falling at least this cutoff from the closest neighbour
kde_cutoff=0.5
gaussian_cutoff=0.35
mle_cutoff=0.66		   # MLE: discard outliers found in the right 1 - mle_cutoff tail area of the distance distribution
bin_size=5                 # Bin size for distance distribution plots
option=$2		   # Option to run pyf2f (see README.md) all or spt,warping,segment,kde,gaussian,mle


# Run pyf2f
python3 Pyf2f_main.py -d ${directory} \
								  --px_size ${px_size} \
								  --particle_diameter ${diameter} \
								  --percentile ${percentile} \
								  --max_displacement ${max_displacement} \
								  --contour_cutoff ${contour_cutoff} \
								  --neigh_cutoff ${neigh_cutoff} \
								  --kde_cutoff ${kde_cutoff} \
								  --gaussian_cutoff ${gaussian_cutoff} \
								  --mle_cutoff ${mle_cutoff} \
								  --dirty \
								  --bin_size ${bin_size}\
								  --verbose --option ${option} >> ${directory}/log.out >& ${directory}/log.err &    
