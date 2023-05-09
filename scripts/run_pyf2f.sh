#!/usr/bin/bash

# PARAMETERS 
directory=$1               # name of your sample directory
px_size=110                # pixel size in nm (Zyla: 65; Prime: 110)
rolling_ball=41            # Rolling ball radius
median_filter=6            # Median filter radius
diameter=7		   # spot detection: diameter of spots in px
percentile=99.7 	   # spot detection: sort spots below this percentile of intensity (select only the brightest spots)
max_displacement=1 	   # linking: link spots from W1-W2 channels separated by a 'max_displacement' in px
contour_cutoff=7	   # segmentation: select spots falling no far from this cutoff to the cell contour
neigh_cutoff=6	  	   # segmentation: select spots falling at least this cutoff from the closest neighbour
kde=0.5                    # KDE cutoff
gaussian=0.35              # R^2 for goodness of gaussian fit
reject_lower=0             # Reject high scores of MLE under this distance estimate
mle_cutoff=0.66		   # MLE: discard outliers found in the right 1 - mle_cutoff tail area of the distance distribution
option="all"		   # Option to run pyf2f (see README.md)


# Run pyf2f
python3 measure_pict_distances.py -d ../${directory}/ \
								  --px_size ${px_size} \
								  --rolling_ball_radius ${rolling_ball} \
								  --median_filter ${median_filter} \
								  --particle_diameter ${diameter} \
								  --percentile ${percentile} \
								  --max_displacement ${max_displacement} \
								  --contour_cutoff ${contour_cutoff} \
								  --neigh_cutoff ${neigh_cutoff} \
								  --kde_cutoff ${kde} \
								  --gaussian_cutoff ${gaussian} \
								  --reject_lower ${reject_lower} \
								  --mle_cutoff ${mle_cutoff} \
								  --option ${option} 1> ../${directory}/${directory}.out 2> ../${directory}/${directory}.err &    
