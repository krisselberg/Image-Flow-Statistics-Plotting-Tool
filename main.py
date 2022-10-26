import imageflowstatplots

# Example of comparing the Middlebury and MPI-Sintel datasets with our library

# data_paths should contain "flow" folder with .flo files and "data" folder with .png images (frames)
# Returns list of CustomDataset objects that support __len__ and __getitem__ functions that return the number of frames in a dataset and the frame at a given index, respectively
datasets = imageflowstatplots.load_datasets(data_paths=['Middlebury/training', 'MPI-Sintel-complete/training/'])

# prints number of frames in Middlebury dataset
print(len(datasets[0]))

# prints number of frames in MBI-Sintel dataset
print(len(datasets[1]))

# prints first frame of Middlebury dataset
print(datasets[0][0])

# prints number of flow matrices in MBI-Sintel dataset
print(len(datasets[1].get_flows()))

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

# Set labels and load in image paths
# luminance_sample - randomly sample [luminance_sample] pixels from each frame for the luminance plot (if not set, uses all pixels )
labels = ["Middlebury", "Sintel"]
colors = ["blue", "red"]
plot_file_name = "my_plot"
luminance_sample = 100000
imageflowstatplots.plot_stats(datasets, labels, colors, plot_file_name, luminance_sample)