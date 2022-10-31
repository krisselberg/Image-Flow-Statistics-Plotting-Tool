import plotter
import io
import numpy as np
import glob

# Demo of comparing the Middlebury and MPI-Sintel datasets with our library

# Data loader to read in .flo files
def read_flo_file(filename, memcached=False):
    """
    Read from .flo file
    :param flow_file: name of the flow file
    :return: optical flow data in matrix
    """
    if memcached:
        filename = io.BytesIO(filename)
    f = open(filename, 'rb')
    magic = np.fromfile(f, np.float32, count=1)[0]
    data2d = None

    if 202021.25 != magic:
        print('Magic number incorrect. Invalid .flo file')
    else:
        w = np.fromfile(f, np.int32, count=1)[0]
        h = np.fromfile(f, np.int32, count=1)[0]
        data2d = np.fromfile(f, np.float32, count=2 * w * h)
        # reshape data into 3D array (columns, rows, channels)
        data2d = np.resize(data2d, (h, w, 2))
    f.close()
    return data2d

# datasets_count - number of datasets
# total_frame_paths and total_flow_matrices contain the .png frame paths and flow matrices for each dataset, respectively
datasets_count = 2
total_frame_paths = [[], []]
total_flow_matrices = [[], []]

frames_dirs = ['data/Middlebury/other-data/**/*.png', 'data/Sintel/MPI-Sintel-training_images/training/final/**/*.png']
flow_dirs = ['data/Middlebury/other-gt-flow/**/*.flo', 'data/Sintel/MPI-Sintel-training_extras/training/flow/**/*.flo']

# populating frame_paths and flow_matrices
for i in range(datasets_count):
    for filename in glob.iglob(frames_dirs[i], recursive = True):
        total_frame_paths[i].append(filename)
    for filename in glob.iglob(flow_dirs[i], recursive = True):
        total_flow_matrices[i].append(read_flo_file(filename))
        
# Plotting the graphs

# Call plot_stats function which saves a figure of image and optical flow statistics plots:
# 3 Image Statistics Plots:
# All image statistics were computed on a grayscale representation of images (each pixel: [0, 255])

# 1. Luminance Histogram

# Calculation - Calculate fraction of all pixels at each luminance value for each dataset. 
# If more than one dataset, we calculate the Kullback-Leibler-Divergence of each luminance distribution to one another.
# Then, plot the results.

# Interpretation - The higher the KL-Divergence is between two datasets, the more different their luminance distributions are.
# This can be useful when comparing how similar an artificially created dataset is to a dataset with natural scenes in terms of luminance.

# 2. Power Spectrum

# Calculation - We calculate the 2D FFT of the 380 x 380 pixel patch in the center of each frame 
# (we subtract the mean intensity of this cropped patch and then apply the Hamming window before to avoid leakage in spectral transformation). 
# Then, we calculate the azimuthally averaged (across all orientations) 1D power spectrum from the 2D power spectrum. 
# We then averaged across all frames in a dataset to get the average power spectrum.
# Lastly, we computed the slope using linear least squares fitting in the log-log space in the range 0 < f ≤ 0.35 cycles/pixelplot
# (since above this range aliasing and pixelation artifacts dominate)
# Linear fits are shown as dashed lines in the figure
# FOR CLARITY, when given multiple datasets, the second and/or third graphs will be shifted by 10**2 and 10**4, respectively.

# Interpretation - Natural scene movies usually exhibit an approximately linear decrease in power with frequency
# (in log-log) with a power spectrum slope around −2 (equivalent to a 1/f 2 falloff).
# Therefore, a power spectrum slope around -2 in the figure indicates that our dataset has images similar to natural scene movies.

# 3. Spatial Derivative

# Calculation - Calculated first difference in the x-direction to plot the log histogram for the horizontal spatial derivative (range from -180 to 180). 
# Calculate the Kurtosis of each distribution (data sets with high kurtosis tend to have heavy tails, or outliers. 
# Data sets with low kurtosis tend to have light tails, or lack of outliers). 

# Interpretation - Datasets with more similar Kurtosis values have image derivatives that match more closely.

# 4 Optical Flow Statistics Plots:
# optical flow is computed using the Classic+NL-fast algorithm (Sun, D., Roth, S., Black, M.J.: Secrets of optical flow estimation and their principles. 
# In: IEEE Conf. on Computer Vision and Pattern Recognition, CVPR. (2010) 2432–2439) using the default parameters.
#
# 1. Motion (u)
#
# Interpretation - We can compare the distribution of large and short motions of datasets.
#
# 2. Speed
#
# Calculation - Speed is defined as the square root of the sum of the squares of u(x, y) and v(x, y)
#
# Interpretation - We can compare how much large motion is present in a dataset.
#
# 3. Flow Direction
#
# Calculation - Direction is defined as the inverse tangent of (v(x, y) / u(x, y))
#
# Interpretation - Natural scenes tend to have broad peaks around 0 and 180 degrees and smaller at +90 and -90 degrees.
# We can compare the similarity of datasets' flow statistics to the flow statistics of natural scenes.
# 
# 4. Spatial Flow Derivative
#
# Calculation - Spatial derivative calculated with first differences of u(x, y)
#
# Interpretation - We can compare the similarity of datasets' flow statistics to the flow statistics of natural scenes.

# labels - labels that will show up for the legend on the plots
# colors - colors that will be used in the plots
# file_limit - extracts first {file_limit} .flo files and .png frames to be used in plotting 
# luminance_sample - randomly sample [luminance_sample] pixels from each frame for the luminance plot (if not set, uses all pixels )
labels = ["Middlebury", "Sintel"]
colors = ["blue", "red"]
file_limit = 1000
luminance_sample = 100000
plotter.plot_stats(total_frame_paths, total_flow_matrices, labels, colors, file_limit, luminance_sample)