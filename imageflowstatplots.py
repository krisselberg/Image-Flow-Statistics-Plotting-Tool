import matplotlib.pyplot as plt
from PIL import Image
import glob
import os
import io
import scipy
import numpy as np
import math
import radialProfile

# CustomDataset should return:
# image path
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

class CustomDataset():
  def __init__(self, data_path):
    # get paths to frames and .flo files
    self.frames_dir = os.path.join('data', data_path, 'frames', "**/*.png")
    self.flo_dir = os.path.join('data', data_path, 'flow', "**/*.flo")
    
    self.all_frame_paths = []
    self.all_flo_paths = []
    for filename in glob.iglob(self.frames_dir, recursive = True):
        self.all_frame_paths.append(filename)
    for filename in glob.iglob(self.flo_dir, recursive = True):
        self.all_flo_paths.append(filename)
    self.flows = []
    for filename in self.all_flo_paths:
        self.flows.append(read_flo_file(filename))

  def get_flows(self):
    return self.flows

  def __len__(self):
    return len(self.all_frame_paths)

  def __getitem__(self, idx):
    # returns a list of the data path at index "idx"
    return self.all_frame_paths[idx]

def load_datasets(data_paths):
    datasets = []
    for data in data_paths:
        datasets.append(CustomDataset(data))
    return datasets

# Luminance

# returns a numpy nd array of 256 numbers, 
# each corresponding to the fraction of pixels in frames of 
# a dataset at a luminance (index)
def get_luminance(dataset, bins, luminance_sample):
    total_lum = np.zeros(len(bins) - 1)

    for idx in range(len(dataset)):
        img = np.asarray(Image.open(dataset[idx]).convert('L')).flatten()
        if luminance_sample and luminance_sample < len(img):
            img = img[np.random.choice(img, size=luminance_sample, replace=False)]
        hist, _ = np.histogram(img, bins)
        total_lum += hist
    fractional_lum = total_lum / np.sum(total_lum)
        
    # delete first element from bins for plotting purposes
    bins = np.delete(bins, 0)
    return fractional_lum, bins
    
def plot_luminance(datasets, labels, colors, axs, luminance_sample):
    print("Creating luminance histogram...")
    lums = []
    for i in range(len(datasets)): 
        bins = np.arange(0, 256, 1)
        fractional_lum, bins = get_luminance(datasets[i], bins, luminance_sample)
        lums.append(fractional_lum)
        axs[0][0].plot(bins, fractional_lum, label=labels[i], color=colors[i])
    # KL Divergence with e as log base
    if len(datasets) == 2:
        kl = scipy.stats.entropy(lums[0], lums[1])
        s = "KL-D {} to {}: {:.3f}".format(labels[0], labels[1], kl)
        axs[0][0].text(0.5, 0.75, s, horizontalalignment='center',
        verticalalignment='center',
        transform=axs[0][0].transAxes)
    axs[0][0].set_title("Luminance Histogram")
    axs[0][0].set_xlabel("Luminance")
    axs[0][0].set_ylabel("Fraction of Pixels")
    axs[0][0].legend()

# Spatial Power Spectra

# center crop is of size 380 x 380 (change as necessary for your dataset)
# (modified from original paper of 436 x 436 because Middlebury has images with smaller dimensions)

def crop_center(img,cropx,cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]

def get_spatial_power_spectra(dataset):
    running_ps = []
    cropx, cropy = 380, 380
    img = np.asarray(Image.open(dataset[0]).convert("L"))
    if img.shape[0] < 380 or img.shape[1] < 380:
        min_dim = min(img.shape[0], img.shape[1])
        cropx, cropy = min_dim, min_dim
    for frame in range(len(dataset)):
        img = Image.open(dataset[frame]).convert("L")
        img = np.asarray(img)
        cropped_img = np.copy(crop_center(img, cropx, cropy))

        # calculate weights of Hamming window
        rmax = 218
        w = np.empty(cropped_img.shape)
        for x in range(cropped_img.shape[1]):
            for y in range(cropped_img.shape[0]):
                r = ((x - 218)**2 + (y - 218)**2)**0.5
                if r < rmax:
                    w_r = 0.54 - 0.46*math.cos(math.pi*(1 - r/rmax))
                else:
                    w_r = 0

                w[x][y] = w_r

        # to avoid leakage in spectral transformation, we subtract the 
        # weighted mean intensity (normalizing_factor) before applying
        # the Hamming window
        normalizing_factor = np.sum(np.multiply(cropped_img, w)) / (np.sum(w))
        normalized_img = (cropped_img - normalizing_factor) / normalizing_factor

        # apply window
        windowed_img = np.multiply(normalized_img, w)

        # Fourier transform to get power spectrum
        # used this paper to calculate: 
        # van der Schaaf, A., van Hateren, J.: Modelling the power spectra of natural images: Statistics and
        # information. Vision Research 36 (1996) 2759 – 2770
        f = np.fft.fftshift(np.fft.fft2(windowed_img))
        ps2d = np.square(np.abs(f))
        ps1d = radialProfile.azimuthalAverage(ps2d)
        running_ps.append(ps1d)

    avg_ps = np.mean(running_ps, axis=0)
    freqs = np.fft.fftfreq(len(avg_ps))

    return freqs, avg_ps

def plot_spatial_power_spectra(datasets, labels, colors, axs):
    print("Creating power spectra plot...")
    for i in range(len(datasets)):
        freqs, avg_ps = get_spatial_power_spectra(datasets[i])
        freqs_cutout = freqs[(freqs > 0) & (freqs <= 0.35)]
        ps_cutout = avg_ps[(freqs > 0) & (freqs <= 0.35)]
        if i > 0:
            ps_cutout *= (10 ** (2 * i))
        a, b = np.polyfit(np.log(freqs_cutout), np.log(ps_cutout), 1)
        y_log = a * np.log(freqs_cutout) + b
        y = np.exp(y_log)
        
        axs[0][1].loglog(freqs_cutout, ps_cutout, color=colors[i], label='{} Slope: {:.3f}'.format(labels[i], a))
        axs[0][1].loglog(freqs_cutout, y, '--', color=colors[i])
        axs[0][1].set_title("Power Spectrum")
        axs[0][1].set_xlabel("Frequency (cycles per pixel)")
        axs[0][1].set_ylabel("log10 (Power)")
        axs[0][1].legend()

# Spatial Derivative (in x-direction)

def get_spatial_derivatives(dataset, bins):
    # min difference can be (0 - 255) and max difference can be (255 - 0)
    total_differences = np.zeros(len(bins) - 1)
    for idx in range(len(dataset)):
        img = np.asarray(Image.open(dataset[idx]).convert('L'), dtype='int32')
        diff = np.diff(img)
        hist, _ = np.histogram(diff, bins)
        total_differences += hist
        
    totalPixelCount = sum(total_differences)
    total_differences = total_differences / totalPixelCount
        
    # delete first element from bins for plotting purposes and cutout values less than 180 and after 180
    bins = np.delete(bins, 0)
    diffs_cutout = total_differences[total_differences != 0]
    bins_cutout = bins[total_differences != 0]
    k = scipy.stats.kurtosis(diffs_cutout)
    # take log for better visualization
    log_diffs_cutout = np.log(diffs_cutout, out=np.zeros_like(diffs_cutout), where=(diffs_cutout != 0))
        
    return log_diffs_cutout, bins_cutout, k

def plot_spatial_derivatives(datasets, labels, colors, axs):
    print("Creating spatial derivative plot...")
    for i in range(len(datasets)): 
        bins = np.arange(-255, 255, 1)
        log_diffs_cutout, bins_cutout, k = get_spatial_derivatives(datasets[i], bins)

        axs[0][2].plot(bins_cutout, log_diffs_cutout, label="{} Kurtosis: {:.2f}".format(labels[i], k), color=colors[i])
        
    axs[0][2].set_title("Spatial Derivative")
    axs[0][2].set_xlabel("dI/dx")
    axs[0][2].set_ylabel("log(fraction of pixels)")
    axs[0][2].legend()

# Motion (u)
def get_motion(flow_matrices, bins):
    total_motion = np.zeros(len(bins) - 1)
    for idx in range(len(flow_matrices)):
        hist, _ = np.histogram(flow_matrices[idx][:,:,0], bins)
        total_motion += hist
        
    totalPixelCount = sum(total_motion)
    total_motion = total_motion / totalPixelCount
        
    # delete first element from bins for plotting purposes
    bins = np.delete(bins, 0)
    motion_cutout = total_motion[total_motion != 0]
    bins_cutout = bins[total_motion != 0]
    # take log for better visualization
    log_motion_cutout = np.log(motion_cutout, out=np.zeros_like(motion_cutout), where=(motion_cutout != 0))
        
    return log_motion_cutout, bins_cutout

def plot_motion(datasets, labels, colors, axs):
    print("Creating optical flow motion plot...")
    for i in range(len(datasets)): 
        bins = np.arange(-300, 300, 1)
        flow_matrices = datasets[i].get_flows()
        log_motion_cutout, bins_cutout = get_motion(flow_matrices, bins)

        axs[1][0].plot(bins_cutout, log_motion_cutout, label="{}".format(labels[i]), color=colors[i])
        
    axs[1][0].set_title("Motion (u)")
    axs[1][0].set_xlabel("u (pixels/frame)")
    axs[1][0].set_ylabel("log(fraction of pixels)")
    axs[1][0].legend()
    
# Speed
def get_speed_direction(flow_matrices, speed_bins, dir_bins):
    total_speed = np.zeros(len(speed_bins) - 1)
    total_dir = np.zeros(len(dir_bins) - 1)
    for idx in range(len(flow_matrices)):
        current_flow = flow_matrices[idx][:,:]
        current_speed = np.sqrt(np.square(current_flow[0]) + np.square(current_flow[1]))
        current_dir = np.degrees(np.arctan2(current_flow[1], current_flow[0]))
        speed_hist, _ = np.histogram(current_speed, speed_bins)
        dir_hist, _ = np.histogram(current_dir, dir_bins)
        total_speed += speed_hist
        total_dir += dir_hist
        
    speed_total_pixel_count = sum(total_speed)
    dir_total_pixel_count = sum(total_dir)
    total_speed = total_speed / speed_total_pixel_count
    total_dir = total_dir / dir_total_pixel_count
        
    # delete first element from bins for plotting purposes
    speed_bins = np.delete(speed_bins, 0)
    dir_bins = np.delete(dir_bins, 0)
    speed_cutout = total_speed[total_speed != 0]
    dir_cutout = total_dir[total_dir != 0]
    speed_bins_cutout = speed_bins[total_speed != 0]
    dir_bins_cutout = dir_bins[total_dir != 0]
    # take log for better visualization
    log_speed_cutout = np.log(speed_cutout, out=np.zeros_like(speed_cutout), where=(speed_cutout != 0))
    log_dir_cutout = np.log(dir_cutout, out=np.zeros_like(dir_cutout), where=(dir_cutout != 0))
        
    return log_speed_cutout, log_dir_cutout, speed_bins_cutout, dir_bins_cutout

def plot_speed_direction(datasets, labels, colors, axs):
    print("Creating optical flow speed and direction plots...")
    for i in range(len(datasets)): 
        speed_bins = np.arange(0, 300, 1)
        dir_bins = np.arange(-180, 180, 1)
        flow_matrices = datasets[i].get_flows()
        log_speed_cutout, log_dir_cutout, speed_bins_cutout, dir_bins_cutout = get_speed_direction(flow_matrices, speed_bins, dir_bins)

        axs[1][1].plot(speed_bins_cutout, log_speed_cutout, label="{}".format(labels[i]), color=colors[i])
        axs[1][2].plot(dir_bins_cutout, log_dir_cutout, label="{}".format(labels[i]), color=colors[i])
        
    axs[1][1].set_title("Speed")
    axs[1][1].set_xlabel("Speed (pixels/frame)")
    axs[1][1].set_ylabel("log(fraction of pixels)")
    axs[1][1].legend()
    axs[1][2].set_title("Flow Direction \u03F4")
    axs[1][2].set_xlabel("\u03F4 (degrees)")
    axs[1][2].set_ylabel("log(fraction of pixels)")
    axs[1][2].legend()

# Spatial flow derivative in x-direction
def get_spatial_flow_derivative(flow_matrices, bins):
    total_differences = np.zeros(len(bins) - 1)
    for idx in range(len(flow_matrices)):
        current_flow = flow_matrices[idx][:,:,0]
        diff = np.diff(current_flow)
        hist, _ = np.histogram(diff, bins)
        total_differences += hist
        
    totalPixelCount = sum(total_differences)
    total_differences = total_differences / totalPixelCount
        
    # delete first element from bins for plotting purposes
    bins = np.delete(bins, 0)
    diffs_cutout = total_differences[total_differences != 0]
    bins_cutout = bins[total_differences != 0]
    # take log for better visualization
    log_diffs_cutout = np.log(diffs_cutout, out=np.zeros_like(diffs_cutout), where=(diffs_cutout != 0))
        
    return log_diffs_cutout, bins_cutout

def plot_spatial_flow_derivative(datasets, labels, colors, axs):
    print("Creating spatial flow derivative plot...")
    for i in range(len(datasets)): 
        bins = np.arange(-100, 100, 1)
        flow_matrices = datasets[i].get_flows()
        log_diffs_cutout, bins_cutout = get_spatial_flow_derivative(flow_matrices, bins)

        axs[1][3].plot(bins_cutout, log_diffs_cutout, label="{}".format(labels[i]), color=colors[i])
        
    axs[1][3].set_title("Spatial Flow Derivative")
    axs[1][3].set_xlabel("du/dx (1/frame)")
    axs[1][3].set_ylabel("log(fraction of pixels)")
    axs[1][3].legend()

def plot_stats(datasets, labels, colors, plot_file_name, luminance_sample=0):
    # Setting up figure
    fig, axs = plt.subplots(2, 4)
    fig.delaxes(axs[0][3])
    
    plot_luminance(datasets, labels, colors, axs, luminance_sample)
    plot_spatial_power_spectra(datasets, labels, colors, axs)
    plot_spatial_derivatives(datasets, labels, colors, axs)
    plot_motion(datasets, labels, colors, axs)
    plot_speed_direction(datasets, labels, colors, axs)
    plot_spatial_flow_derivative(datasets, labels, colors, axs)

    fig.set_size_inches(16, 12)
    # when saving, specify the DPI
    plt.savefig(plot_file_name + ".png", dpi = 100)



# Questions:
# Cropping error