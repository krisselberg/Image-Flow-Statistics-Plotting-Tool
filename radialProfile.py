import numpy as np
# Contributed by Jessica R. Lu
# A helper function to create the azimuthally averaged 1D radial profile from a 2D image. Right now the code is very crude (calculates the center of the image itself, assumes a bin size of 1, etc.). But as I (or other people) improve the code, I will keep the version below updated. Dependencies include
# https://www.astrobetter.com/blog/2010/03/03/fourier-transforms-of-images-in-python/
# https://www.astrobetter.com/wiki/tiki-index.php?page=python_radial_profiles
# numpy
# radialProfile.py I've borrowed & modified radialprofile.py - the above version excludes the inner & outer points. I also included the ability to measure the radial standard deviation.  

# Contributed by Adam Ginsburg
# https://github.com/keflavich/image_tools/blob/master/image_tools/radialprofile.py

# Contributed by Ian J. Crossfield
# Here's my own version of such a function, with some of the desired options above (user-specified centering, custom bin sizes) already included. radial_data.py

# used to calculate azimuthally averaged 1D power spectrum from 2D power spectrum
def azimuthalAverage(image, center=None):
    """
    Calculate the azimuthally averaged radial profile.

    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is 
             None, which then uses the center of the image (including 
             fracitonal pixels).
    
    """
    # Calculate the indices from the image
    y, x = np.indices(image.shape)

    if not center:
        center = np.array([(x.max()-x.min())/2.0, (x.max()-x.min())/2.0])

    r = np.hypot(x - center[0], y - center[1])

    # Get sorted radii
    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind]
    i_sorted = image.flat[ind]

    # Get the integer part of the radii (bin size = 1)
    r_int = r_sorted.astype(int)

    # Find all pixels that fall within each radial bin.
    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
    rind = np.where(deltar)[0]       # location of changed radius
    nr = rind[1:] - rind[:-1]        # number of radius bin
    
    # Cumulative sum to figure out sums for each radius bin
    csim = np.cumsum(i_sorted, dtype=float)
    tbin = csim[rind[1:]] - csim[rind[:-1]]

    radial_prof = tbin / nr

    return radial_prof
