# Image and Optical Flow Statistics Plotting Tool
This repository contains visualization tools for plotting image and optical flow statistics (luminance, power spectrum, spatial derivatives, motion, speed, and flow direction)
from optical flow datasets containing frames and .flo files.

# Requirements
This code has been tested with Python 3.7.6 and Pillow 9.2.0.

    conda env create -f environment.yaml
    conda activate plotenv
    
# Demo
To download the MBI-Sintel and Middlebury datasets for the demos, run

    chmod ug+x download_datasets.sh && ./download_datasets.sh
    
By default plotter.py will search for the .png and .flo files in these locations:

    ├── data
        ├── * [data_path]
            ├── frames
                ├── **/*.png
            ├── flow
                ├── **/*.flo
        ├── * [data_path]
            ├── frames
                ├── **/*.png
            ├── flow
                ├── **/*.flo
