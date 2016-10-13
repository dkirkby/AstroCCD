# Licensed under a 3-clause BSD style license - see LICENSE file.
"""Utilities for working with numerical sensor data in numpy arrays.
"""
from __future__ import print_function, division

import numpy as np
import numpy.lib.stride_tricks
import scipy.stats


def block_view(A, block_shape):
    """Provide a 2D block view of a 2D array.

    Returns a view with shape (n, m, a, b) for an input 2D array with
    shape (n*a, m*b) and block_shape of (a, b).
    """
    assert len(A.shape) == 2, '2D input array is required.'
    assert A.shape[0] % block_shape[0] == 0, \
        'Block shape[0] does not evenly divide array shape[0].'
    assert A.shape[1] % block_shape[1] == 0, \
        'Block shape[1] does not evenly divide array shape[1].'
    shape = (A.shape[0] / block_shape[0], A.shape[1] / block_shape[1]) + block_shape
    strides = (block_shape[0] * A.strides[0], block_shape[1] * A.strides[1]) + A.strides
    return numpy.lib.stride_tricks.as_strided(A, shape=shape, strides=strides)


def apply_filter(A, smoothing, power=2.0):
    """Apply a hi/lo pass filter to a 2D image.

    The value of smoothing specifies the cutoff wavelength in pixels,
    with a value >0 (<0) applying a hi-pass (lo-pass) filter. The
    lo- and hi-pass filters sum to one by construction.  The power
    parameter determines the sharpness of the filter, with higher
    values giving a sharper transition.
    """
    if smoothing == 0:
        return A
    ny, nx = A.shape
    # Round down dimensions to even values for rfft.
    # Any trimmed row or column will be unfiltered in the output.
    nx = 2 * (nx // 2)
    ny = 2 * (ny // 2)
    T = np.fft.rfft2(A[:ny, :nx])
    # Last axis (kx) uses rfft encoding.
    kx = np.fft.rfftfreq(nx)
    ky = np.fft.fftfreq(ny)
    kpow = (kx ** 2 + ky[:, np.newaxis] ** 2) ** (power / 2.)
    k0pow = (1. / smoothing) ** power
    if smoothing > 0:
        F = kpow / (k0pow + kpow) # high pass
    else:
        F = k0pow / (k0pow + kpow) # low pass
    S = A.copy()
    S[:ny, :nx] = np.fft.irfft2(T * F)
    return S


def zero_by_region(data, region_shape, num_sigmas_clip=4.0, smoothing=250, power=4):
    """Subtract the clipped median signal in each amplifier region.

    Optionally also remove any smooth variation in the mean signal with
    a high-pass filter controlled by the smoothing and power parameters.
    Returns a an array of median levels in each region and a mask of
    unclipped pixels.
    """
    mask = np.zeros_like(data, dtype=bool)

    # Loop over amplifier regions.
    regions = block_view(data, region_shape)
    masks = block_view(mask, region_shape)
    ny, nx = regions.shape[:2]
    levels = np.empty((ny, nx))

    for y in range(ny):
        for x in range(nx):
            region_data = regions[y, x]
            region_mask = masks[y, x]
            clipped1d, lo, hi = scipy.stats.sigmaclip(
                region_data, num_sigmas_clip, num_sigmas_clip)
            # Add unclipped pixels to the mask.
            region_mask[(region_data > lo) & (region_data < hi)] = True
            # Subtract the clipped median in place.
            levels[y, x] = np.median(clipped1d)
            region_data -= levels[y, x]
            # Smooth this region's data.
            if smoothing != 0:
                clipped_data = region_data[~region_mask]
                region_data[~region_mask] = 0.
                region_data[:] = apply_filter(region_data, smoothing, power)
                region_data[~region_mask] = clipped_data

    return levels, mask


def equalize(A, clip_percent=5):
    """Equalize the values of an array.

    The returned array has values between 0-1 such that clip_percent
    of the values are clipped symmetrically at 0 and 1, and the
    histogram of values between 0 and 1 is flat. This is a non-linear
    transformation and primarily useful for showing small variations
    over a large dynamic range.
    """
    A_flat = A.reshape(-1)
    n = len(A_flat)
    num_clip = round(n * clip_percent / 100.)
    num_clip_lo = num_clip // 2
    num_clip_hi = num_clip - num_clip_lo
    equalized = np.empty_like(A_flat, dtype=float)
    order = np.argsort(A_flat)
    equalized[order] = np.clip(
        (np.arange(n) - num_clip_lo) / float(n - num_clip), 0., 1.)
    return equalized.reshape(A.shape)


def downsample(data, mask, downsampling):
    """Downsample a masked image by the specified integer factor.

    The returned data is the average of unmasked input pixel values
    within each downsampled block.  The returned mask selects downsampled
    blocks with at least one unmasked input pixel.
    """
    downsampling = int(downsampling)
    assert downsampling > 0, 'Expected downsampling > 0.'
    npix = block_view(mask, (downsampling, downsampling)).sum(axis=(2, 3))
    mask = npix > 0
    data = block_view(data, (downsampling, downsampling)).sum(axis=(2, 3))
    data[mask] /= npix[mask]
    return data, mask


def measure_profile(data, mask, line_start, line_stop, sigma=50, num_bins=50):
    """Measure a 1D profile along a line through 2D data.

    The sigma parameter determines the effective width of the measurement line
    in units of pixels, and is used to apply Gaussian weights as a function of
    transverse distance from the line. Returns the profile bin edges and values.
    """
    x1, y1 = line_start
    x2, y2 = line_stop
    dx, dy = x2 - x1, y2 - y1
    dr = np.sqrt(dx ** 2 + dy ** 2)

    # Calculate the rotation angle that puts the line along +x.
    th = np.arctan2(dy, dx)
    sin_th, cos_th = np.sin(th), np.cos(th)

    # Calculate pixel distances relative to the line.
    ny, nx = data.shape
    x, y = np.arange(nx), np.arange(ny)[:, np.newaxis]
    u = (cos_th * (x - x1) + sin_th * (y - y1))
    v = (-sin_th * (x - x1) + cos_th * (y - y1))

    # Use Gaussian weighting in the transverse direction.
    wgt = np.exp(-0.5 * (v / sigma) ** 2)
    M = mask & (u >= 0) & (u <= dr)

    # Calculate the 1D profile histogram.
    D = u[M].reshape(-1)
    W1 = data[M].reshape(-1)
    W2 = wgt[M].reshape(-1)
    profile, edges = np.histogram(D, bins=num_bins, range=(0, dr), weights=W1 * W2)
    wsum, edges = np.histogram(D, bins=num_bins, range=(0, dr), weights=W2)
    profile /= wsum

    return edges, profile
