#!/usr/bin/bash

# PARAMETERS
directory=$1
reg_input=$2               # name of input directory for creating registration map
test_input=$3              # name of input directory for testing registration
reg_output=$4              # name of output directory for saving map coordinates
test_output=$5             # name of output directory for saving TRE
px_size=64.5               # pixel size in nm (Zyla: 65; Prime: 110)
diameter=11		           # spot detection: diameter of spots in px
percentile=99.8 	       # spot detection: sort spots below this percentile of intensity (select only the brightest spots)
max_displacement=2 	       # linking: link spots from W1-W2 channels separated by a 'max_displacement' in px

# Run pyf2f
python3 PyF2F_image_registration.py -bri ${directory}/${reg_input}/ \
                                  -bti ${directory}/${test_input}/ \
                                  -bro ${directory}/${reg_output}/ \
                                  -bto ${directory}/${test_output}/ -px ${px_size} \
                                  -sd ${diameter} \
				  -pc ${percentile} \
				  -l ${max_displacement} \
                                  --global
								 
