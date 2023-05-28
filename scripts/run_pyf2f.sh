#!/usr/bin/bash

# PARAMETERS 
directory=$1           # name of your sample directory
px_size=64.5           # pixel size in nm (Zyla: 65; Prime: 110)
rb_radius=70           # Rolling ball radius
md_radius=10		   # Median filter radius
diameter=11			   # spot detection: diameter of spots in px
percentile=99.7		   # spot detection: sort spots below this percentile of intensity (select only the brightest spots)
max_displacement=2 	   # linking: link spots from W1-W2 channels separated by a 'max_displacement' in px
contour_cutoff=13	   # segmentation: select spots falling no far from this cutoff to the cell contour
neigh_cutoff=9	  	   # segmentation: select spots falling at least this cutoff from the closest neighbour
mle_cutoff=0.66		   # MLE: discard outliers found in the right 1 - mle_cutoff tail area of the distance distribution
option="all"		   # Option to run pyf2f (see README.md)


# Run pyf2f
python3 Pyf2f_main.py -d ../${directory}/ \
								  --px_size ${px_size} \
								  --particle_diameter ${diameter} \
								  --percentile ${percentile} \
								  --max_displacement ${max_displacement} \
								  --contour_cutoff ${contour_cutoff} \
								  --neigh_cutoff ${neigh_cutoff} \
								  --mle_cutoff ${mle_cutoff} \
								  --verbose \
								  --dirty \
								  --option ${option} >> ../${directory}/${directory}.out >& ../${directory}/${directory}.err &    
