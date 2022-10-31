# Image and Optical Flow Statistics Plotting Tool
This repository contains visualization tools for plotting image and optical flow statistics (luminance, power spectrum, spatial derivatives, motion, speed, and flow direction) from optical flow datasets.

# Requirements
This code has been tested with Python 3.7.6 and Pillow 9.2.0.

    conda env create -f environment.yaml
    conda activate plotenv
    
# Demo
To download the [MBI-Sintel](http://sintel.is.tue.mpg.de/) and [Middlebury](https://vision.middlebury.edu/flow/data/) datasets for the demos, run

    chmod ug+x download_datasets.sh && ./download_datasets.sh
                
To plot all image statistics and optical flow statistics for the downloaded datasets, run

    python demo.py
    
.png files of each plot will be saved under these file names: `luminance.png`, `power_spectra.png`, `spatial_derivative.png`, `motion.png`, `direction.png`, `speed.png`, and `spatial_flow_derivative.png`. This is the expected output from `demo.py`:

<p align="middle">
  <img src="/screenshots/luminance.png" width=250 />
  <img src="/screenshots/power_spectra.png" width=250 />
  <img src="/screenshots/spatial_derivative.png" width=250 />
  <img src="/screenshots/motion.png" width=250 />
  <img src="/screenshots/speed.png" width=250 /> 
  <img src="/screenshots/direction.png" width=250 />
  <img src="/screenshots/spatial_flow_derivative.png" width=250 />
</p>

# Instructions

Modify `main.py` for use with your own datasets.
