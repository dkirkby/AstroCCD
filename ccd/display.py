# Licensed under a 3-clause BSD style license - see LICENSE file.
"""Utilities for working with numerical sensor data in numpy arrays.
"""
from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects
import mpl_toolkits.axes_grid.inset_locator


def plot_pixels(data, lo, hi, cmap='plasma', dpi=150.):
    """Create an image with pixel dimensions that match the data dimensions.
    """
    h, w = data.shape
    figure = plt.figure(figsize=(w/dpi, h/dpi), frameon=False, dpi=dpi)
    axes = plt.Axes(figure, [0., 0., 1., 1.])
    axes.axis(xmin=-0.5, xmax=w-0.5, ymin=-0.5, ymax=h-0.5)
    axes.set_axis_off()
    figure.add_axes(axes)
    plt.imshow(data, interpolation='none', vmin=lo, vmax=hi, cmap=cmap,
               aspect='equal', origin='lower')
    plt.axis('off')
    return figure, axes


def add_label(label, label_pos='tr', label_size=0.025, axes=None):
    """Annotate the current plot with a text label.
    """
    if axes is None:
        axes = plt.gca()
    args = {
        'tr': dict(xy=(0.99, 0.99), xytext=(0.99, 0.99), ha='right', va='top'),
        'tl': dict(xy=(0.01, 0.99), xytext=(0.01, 0.99), ha='left', va='top'),
        'bl': dict(xy=(0.01, 0.01), xytext=(0.01, 0.01), ha='left', va='bottom'),
        'br': dict(xy=(0.01, 0.99), xytext=(0.99, 0.01), ha='right', va='bottom')
    }[label_pos]
    size = label_size * (axes.transAxes.transform((0, 1))[1] -
                         axes.transAxes.transform((0, 0))[1])
    axes.annotate(
        label, xycoords='axes fraction', textcoords='axes fraction',
        fontsize=size, fontweight='bold', color='white', path_effects=[
            matplotlib.patheffects.Stroke(linewidth=1, foreground='black')],
        **args)


def create_inset(axes, location='br', width='40%', height='30%'):
    """Create axes for an inset plot.
    """
    loc = dict(tr=1, tl=2, bl=3, br=4, l=6,
               r=7, b=8, t=9, c=10)
    try:
        loc = loc[location]
    except KeyError:
        raise ValueError('location must be one of {0}.'.format(loc.keys()))
    return mpl_toolkits.axes_grid.inset_locator.inset_axes(
        axes, width, height, loc=loc)


def draw_profile(edges, profile, axes, xgrid=10, ygrid=2):
    """
    Draw a binned profile on the specified axes, using the specified grid
    spacing and without any label outside the plot frame.
    """
    dr = edges[-1]
    centers = 0.5 * (edges[1:] + edges[:-1])
    ymax = 1.1 * np.max(np.abs(profile))
    axes.plot(centers, profile, 'k.-')
    axes.set_xlim(0, dr)
    axes.set_ylim(-ymax, +ymax)
    axes.set_xticks(np.arange(0, dr, xgrid))
    ygrid = np.arange(ygrid, ymax, ygrid)
    axes.set_yticks(np.hstack((-ygrid[::-1], ygrid)))
    axes.grid(c='b', ls='-', alpha=0.5)
    axes.axhline(0, c='r', ls='-', lw=2)
    # This removes the axis labels but not the space they occupied.
    axes.xaxis.set_ticklabels([])
    axes.yaxis.set_ticklabels([])


def draw_line(line_start, line_stop, half_width, axes, scale=1,
              fc='w', ec='k', alpha=0.75, **kwargs):
    """
    Draw a rotated rectangle to represent a line whose midline runs
    from start to stop and has the specified half-width.  Any additional
    kwargs are passed to the Polygon constructor.
    """
    # Calculate the rectangle corners of a rectangle
    x1, y1 = line_start
    x2, y2 = line_stop
    dx, dy = x2 - x1, y2 - y1
    dr = np.sqrt(dx ** 2 + dy ** 2)
    corners = np.array([[0, -half_width], [0, +half_width],
                        [dr, +half_width], [dr, -half_width]])
    th = np.arctan2(dy, dx)
    sin_th, cos_th = np.sin(th), np.cos(th)
    R = np.array([[cos_th, sin_th], [-sin_th, cos_th]])
    corners = corners.dot(R)
    corners += [x1, y1]
    # Add a polygon to the specified axes, scaled down by the specified fraction.
    poly = matplotlib.patches.Polygon(
        corners / scale, fc=fc, ec=ec, alpha=alpha, **kwargs)
    axes.add_patch(poly)
