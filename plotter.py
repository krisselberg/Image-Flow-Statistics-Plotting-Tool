import matplotlib.pyplot as plt
from PIL import Image
import glob
import os
from scipy import stats
import numpy as np
import math
import radialProfile

# Luminance

def get_luminance(frame_paths, bins, luminance_sample):
    total_lum = np.zeros(len(bins) - 1)

    for i in range(len(frame_paths)):
        img = np.asarray(Image.open(frame_paths[i]).convert('L')).flatten()
        if luminance_sample and luminance_sample < len(img):
            img = img[np.random.choice(img, size=luminance_sample, replace=False)]
        hist, _ = np.histogram(img, bins)
        total_lum += hist
    fractional_lum = total_lum / np.sum(total_lum)
        
    # delete first element from bins for plotting purposes
    bins = np.delete(bins, 0)
    return fractional_lum, bins
    
def plot_luminance(total_frame_paths, labels, colors, luminance_sample):
    print("Creating luminance histogram...")
    fig, ax = plt.subplots()
    lums = []
    for i in range(len(total_frame_paths)): 
        bins = np.arange(0, 256, 1)
        fractional_lum, bins = get_luminance(total_frame_paths[i], bins, luminance_sample)
        lums.append(fractional_lum)
        plt.plot(bins, fractional_lum, label=labels[i], color=colors[i])
    # KL Divergence with e as log base
    if len(total_frame_paths) == 2:
        kl = stats.entropy(lums[0], lums[1])
        s = "KL-D {} to {}: {:.3f}".format(labels[0], labels[1], kl)
        plt.text(0.5, 0.75, s, horizontalalignment='center',
        verticalalignment='center',
        transform=ax.transAxes)
    plt.title("Luminance Histogram")
    plt.xlabel("Luminance")
    plt.ylabel("Fraction of Pixels")
    plt.legend()
    plt.savefig("luminance.png", dpi = 100)

# Spatial Power Spectra

# center crop is of size 380 x 380 (change as necessary for your dataset)
# (modified from original paper of 436 x 436 because Middlebury has images with smaller dimensions)

def crop_center(img,cropx,cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]

def get_spatial_power_spectra(frame_paths):
    running_ps = []
    cropx, cropy = 380, 380
    img = np.asarray(Image.open(frame_paths[0]).convert("L"))
    if img.shape[0] < 380 or img.shape[1] < 380:
        min_dim = min(img.shape[0], img.shape[1])
        cropx, cropy = min_dim, min_dim
    for i in range(len(frame_paths)):
        img = Image.open(frame_paths[i]).convert("L")
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

def plot_spatial_power_spectra(total_frame_paths, labels, colors):
    print("Creating power spectra plot...")
    fig, ax = plt.subplots()
    for i in range(len(total_frame_paths)):
        freqs, avg_ps = get_spatial_power_spectra(total_frame_paths[i])
        freqs_cutout = freqs[(freqs > 0) & (freqs <= 0.35)]
        ps_cutout = avg_ps[(freqs > 0) & (freqs <= 0.35)]
        if i > 0:
            ps_cutout *= (10 ** (2 * i))
        a, b = np.polyfit(np.log(freqs_cutout), np.log(ps_cutout), 1)
        y_log = a * np.log(freqs_cutout) + b
        y = np.exp(y_log)
        
        plt.loglog(freqs_cutout, ps_cutout, color=colors[i], label='{} Slope: {:.3f}'.format(labels[i], a))
        plt.loglog(freqs_cutout, y, '--', color=colors[i])
    plt.title("Power Spectrum")
    plt.xlabel("Frequency (cycles per pixel)")
    plt.ylabel("log10 (Power)")
    plt.legend()
    plt.savefig("power_spectra.png", dpi = 100)

# Spatial Derivative (in x-direction)

def get_spatial_derivatives(frame_paths, bins):
    # min difference can be (0 - 255) and max difference can be (255 - 0)
    total_differences = np.zeros(len(bins) - 1)
    for i in range(len(frame_paths)):
        img = np.asarray(Image.open(frame_paths[i]).convert('L'), dtype='int32')
        diff = np.diff(img)
        hist, _ = np.histogram(diff, bins)
        total_differences += hist
        
    totalPixelCount = sum(total_differences)
    total_differences = total_differences / totalPixelCount
        
    # delete first element from bins for plotting purposes and cutout values less than 180 and after 180
    bins = np.delete(bins, 0)
    diffs_cutout = total_differences[total_differences != 0]
    bins_cutout = bins[total_differences != 0]
    k = stats.kurtosis(diffs_cutout)
    # take log for better visualization
    log_diffs_cutout = np.log(diffs_cutout, out=np.zeros_like(diffs_cutout), where=(diffs_cutout != 0))
        
    return log_diffs_cutout, bins_cutout, k

def plot_spatial_derivatives(total_frame_paths, labels, colors):
    print("Creating spatial derivative plot...")
    fig, ax = plt.subplots()
    for i in range(len(total_frame_paths)): 
        bins = np.arange(-255, 255, 1)
        log_diffs_cutout, bins_cutout, k = get_spatial_derivatives(total_frame_paths[i], bins)

        plt.plot(bins_cutout, log_diffs_cutout, label="{} Kurtosis: {:.2f}".format(labels[i], k), color=colors[i])
        
    plt.title("Spatial Derivative")
    plt.xlabel("dI/dx")
    plt.ylabel("log(fraction of pixels)")
    plt.legend()
    plt.savefig("spatial_derivative.png", dpi = 100)

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

def plot_motion(total_flow_matrices, labels, colors):
    print("Creating optical flow motion plot...")
    fig, ax = plt.subplots()
    for i in range(len(total_flow_matrices)): 
        bins = np.arange(-300, 300, 1)
        flow_matrices = total_flow_matrices[i]
        log_motion_cutout, bins_cutout = get_motion(flow_matrices, bins)

        plt.plot(bins_cutout, log_motion_cutout, label="{}".format(labels[i]), color=colors[i])
        
    plt.title("Motion (u)")
    plt.xlabel("u (pixels/frame)")
    plt.ylabel("log(fraction of pixels)")
    plt.legend()
    plt.savefig("motion.png", dpi = 100)
    
# Speed
def get_speed(flow_matrices, bins):
    total_speed = np.zeros(len(bins) - 1)
    for idx in range(len(flow_matrices)):
        current_flow = flow_matrices[idx][:,:]
        current_speed = np.sqrt(np.square(current_flow[0]) + np.square(current_flow[1]))
        speed_hist, _ = np.histogram(current_speed, bins)
        total_speed += speed_hist
        
    total_pixel_count = sum(total_speed)
    total_speed = total_speed / total_pixel_count
        
    # delete first element from bins for plotting purposes
    bins = np.delete(bins, 0)
    speed_cutout = total_speed[total_speed != 0]
    bins_cutout = bins[total_speed != 0]
    # take log for better visualization
    log_speed_cutout = np.log(speed_cutout, out=np.zeros_like(speed_cutout), where=(speed_cutout != 0))
        
    return log_speed_cutout, bins_cutout

def plot_speed(total_flow_matrices, labels, colors):
    print("Creating optical flow speed plots...")
    fig, ax = plt.subplots()

    for i in range(len(total_flow_matrices)): 
        bins = np.arange(0, 300, 1)
        flow_matrices = total_flow_matrices[i]
        log_speed_cutout, bins_cutout = get_speed(flow_matrices, bins)
        plt.plot(bins_cutout, log_speed_cutout, label="{}".format(labels[i]), color=colors[i])
        
    plt.title("Speed")
    plt.xlabel("Speed (pixels/frame)")
    plt.ylabel("log(fraction of pixels)")
    plt.legend()
    plt.savefig("speed.png", dpi = 100)

# Direction
def get_direction(flow_matrices, bins):
    total_dir = np.zeros(len(bins) - 1)
    for idx in range(len(flow_matrices)):
        current_flow = flow_matrices[idx][:,:]
        current_dir = np.arctan2(current_flow[:,:,1], current_flow[:,:,0]) * (180 / np.pi)
        dir_hist, _ = np.histogram(current_dir, bins)
        total_dir += dir_hist
        
    total_pixel_count = sum(total_dir)
    total_dir = total_dir / total_pixel_count
        
    # delete first element from bins for plotting purposes
    bins = np.delete(bins, 0)
    dir_cutout = total_dir[total_dir != 0]
    bins_cutout = bins[total_dir != 0]
    # take log for better visualization
    log_dir_cutout = np.log(dir_cutout, out=np.zeros_like(dir_cutout), where=(dir_cutout != 0))
        
    return log_dir_cutout, bins

def plot_direction(total_flow_matrices, labels, colors):
    print("Creating optical flow direction plots...")
    fig, ax = plt.subplots()

    for i in range(len(total_flow_matrices)): 
        bins = np.arange(-180, 180, 1)
        flow_matrices = total_flow_matrices[i]
        log_dir_cutout, bins = get_direction(flow_matrices, bins)
        plt.plot(bins, log_dir_cutout, label="{}".format(labels[i]), color=colors[i])

    plt.title("Flow Direction \u03F4")
    plt.xlabel("\u03F4 (degrees)")
    plt.ylabel("log(fraction of pixels)")
    plt.legend()
    plt.savefig("direction.png", dpi = 100)

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

def plot_spatial_flow_derivative(total_flow_matrices, labels, colors):
    print("Creating spatial flow derivative plot...")
    fig, ax = plt.subplots()
    for i in range(len(total_flow_matrices)): 
        bins = np.arange(-100, 100, 1)
        flow_matrices = total_flow_matrices[i]
        log_diffs_cutout, bins_cutout = get_spatial_flow_derivative(flow_matrices, bins)

        plt.plot(bins_cutout, log_diffs_cutout, label="{}".format(labels[i]), color=colors[i])
        
    plt.title("Spatial Flow Derivative")
    plt.xlabel("du/dx (1/frame)")
    plt.ylabel("log(fraction of pixels)")
    plt.legend()
    plt.savefig("spatial_flow_derivative.png", dpi = 100)

def plot_stats(total_frame_paths, total_flow_matrices, labels, colors, file_limit=0, luminance_sample=0):
    # if file_limit, extract paths
    if file_limit:
        for i in range(len(total_frame_paths)):
            total_frame_paths[i] = total_frame_paths[i][:file_limit]
            total_flow_matrices[i] = total_flow_matrices[i][:file_limit]
    
    plot_luminance(total_frame_paths, labels, colors, luminance_sample)
    plot_spatial_power_spectra(total_frame_paths, labels, colors)
    plot_spatial_derivatives(total_frame_paths, labels, colors)
    plot_motion(total_flow_matrices, labels, colors)
    plot_speed(total_flow_matrices, labels, colors)
    plot_direction(total_flow_matrices, labels, colors)
    plot_spatial_flow_derivative(total_flow_matrices, labels, colors)