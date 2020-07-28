###############################################################################
# !/usr/bin/env python                                                        #
#  -*- coding: utf-8 -*-                                                      #
#                         (C) Iring Kösters, 2020                             #
#                  Otto-von-Guericke University Magdeburg                     #
###############################################################################
# This program is free software: you can redistribute it and/or modify        #
# it under the terms of the GNU General Public License as published by        #
# the Free Software Foundation, either version 3 of the License, or           #
# (at your option) any later version.                                         #
# This program is distributed in the hope that it will be useful,             #
# but WITHOUT ANY WARRANTY; without even the implied warranty of              #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.                        #
# See the GNU General Public License for more details.                        #
# You should have received a copy of the GNU General Public License           #
# along with this program.                                                    #
# If not, see <http://www.gnu.org/licenses/>.                                 #
###############################################################################

import json
import os

import numpy as np
import pims
from matplotlib import rc
from scipy import ndimage as ndi
from scipy.interpolate import interp1d
from tqdm import trange

from inspectiontools import plot_image
from midline2mask import butter_lowpass_filter


def compute_midline(frame_nb, outline):

    lower_side, upper_side = split_outline(outline)

    # plt.axis("equal")
    # plt.plot(lower_side[:, 0]*-1+850, lower_side[:, 1], "C0", alpha=0.5)
    # plt.plot(upper_side[:, 0]*-1+850, upper_side[:, 1], "C1", alpha=0.5)

    lower_side[:, 1] = butter_lowpass_filter(lower_side[:, 1], 0.005, 1, 5)
    upper_side[:, 1] = butter_lowpass_filter(upper_side[:, 1], 0.005, 1, 5)

    outline = np.vstack((lower_side, upper_side))

    # Compute midline
    midline = fit(outline, 3, np.linspace(50, 620, 1000))

    # Cut midline to the known length tailwise
    midline_len = calc_len(midline)
    idx_piv_mid = find_idx_nearest(midline[:, 0], pivot[0])
    midline_len -= midline_len[idx_piv_mid]
    len_correction = -0.005
    midline = midline[
        np.argwhere(midline_len > chord_len * (-(1 - c_pivot) + len_correction))[:, 0]
    ]

    # Image output
    mid_im = crop(np.ones((2000, 2000)))
    mid_im[midline.astype("uint")[:, 0], midline.astype("uint")[:, 1]] = 0
    mid_im[outline.astype("uint")[:, 0], outline.astype("uint")[:, 1]] = 0
    plot_image(mid_im, "./midline/", "mid", frame_nb, 100)

    # Get values of flexible part of midline
    midline_len = calc_len(midline)
    flex_sample_positions = np.linspace(
        0, 1 - c_flexrigid, 4, endpoint=False
    )  # 4 points for 3rd degree fit
    flex_midline_points = np.zeros((0, 2))
    for sample_p in flex_sample_positions:
        point = midline[find_idx_nearest(midline_len, sample_p * chord_len)]
        flex_midline_points = np.vstack((flex_midline_points, point))

    # Correction of tail tip where first order midline can be assumed
    flex_grads = np.gradient(flex_midline_points, axis=0)
    flex_midline_points[0, :] += flex_grads[0, :] - flex_grads[1, :]

    # Get values of rigid part of midline
    pos = pos_array[frame_nb]
    rigid_sample_positions = np.linspace(
        c_pivot - c_flexrigid, c_pivot, 3, endpoint=True
    )
    rigid_midline_points = np.zeros((0, 2))
    for sample_p in rigid_sample_positions:
        point = np.array(
            (
                np.cos(np.deg2rad(pos)) * sample_p * chord_len + pivot[0],
                np.sin(np.deg2rad(pos)) * sample_p * chord_len + pivot[1],
            )
        )
        rigid_midline_points = np.vstack((rigid_midline_points, point))

    midline_points = np.vstack((flex_midline_points, rigid_midline_points))

    return midline_points


def compute_outline(frame_nb, sum_amount):
    # Segment hydrofoil
    sumframe = sum_frames(frame_nb, sum_amount)

    # Image output
    sumf = np.where(sumframe == 0, 1, 0)
    plot_image(sumf, "./sumframe/", "sum", frame_nb, 100)

    corr = ndi.correlate(sumframe, round_kernel(20))
    corr = np.where(corr == 0, 0, 1)

    # Image output
    plot_image(np.where(corr == 1, 0, 1), "./correlate/", "corr", frame_nb, 100)

    bin_dilate = ndi.binary_closing(corr, round_kernel(10), 2)
    bin_dilate = np.where(bin_dilate == 0, 1, 0)
    struct = get_biggest_structure(bin_dilate)
    struct = ndi.binary_fill_holes(struct, round_kernel(10))

    # Image output
    plot_image(np.where(struct == 0, 0, 1), "./dilate/", "dil", frame_nb, 100)

    outline = ndi.filters.laplace(struct)  # Make 2-Pixel Outline
    outline = np.array(np.where(outline == True)).T  # Extract Outlinevalue Indices
    return outline


def get_file(path="./"):
    """read cinefile in specified path"""
    film = pims.open(path + fn_film)
    return film


def get_frame(frame_number):
    """Extracts a frame from the cine file"""
    frame = film.get_frame(frame_number).T
    return frame


def crop(frame, xstart=300, xend=1150, ystart=0, yend=800):
    """Crops an 2D-array to a wanted size.
    
    Parameters
    ----------
    frame : ndarray
        Array which to crop
    x_args : list, optional
        X-values for start and stop, by default [300, 1150]
    y_args : list, optional
        Y-values for start and stop, by default [200, 650]
    
    Returns
    -------
    cropped_frame: ndarray
        Frame cropped to specified size
    """
    cropped_frame = frame[xstart:xend, ystart:yend]
    return cropped_frame


def denoise(frame, percentile=99):
    """Reduce noise by filtering for values which are above a threshold, 
    determined by a one-sided percentile (no upper, only lower limit). 
    Remaining values are reduced by that that number, to remove offset.
    
    Parameters
    ----------
    frame : ndarray
        Array to denoise
    percentile : float, optional
        Percentile of values which should be above threshold, by default 99
    
    Returns
    -------
    denoised_frame: ndarray, uint16
        Denoised array, values below set to 0 - above: orig. value minus threshold value
    """
    threshold = np.percentile(frame, percentile)
    prepro_frame = np.where(frame < threshold, 0, frame - threshold).astype("uint16")
    return prepro_frame


def sum_frames(start_at, sum_amount):
    accum_frame = np.zeros_like(crop(get_frame(0)))
    for f in range(start_at, start_at + sum_amount):
        frame = crop(get_frame(f))
        frame = denoise(frame)
        frame = np.where(frame > 0, 1, 0)
        accum_frame = accum_frame + frame
    return np.where(accum_frame > 0, 1, 0)


def round_kernel(size):
    """Creates a square array with zeros with a circular shape of ones, which is as large as it can fit the array.
    
    Parameters
    ----------
    size : int
        Length in elements along each axis
    
    Returns
    -------
    out: ndarray
        Square sized array containing only zeros and ones
    """
    a, b, r = size // 2, size // 2, size // 3  # a,b = midpoint; r = radius

    y, x = np.ogrid[-a : size - a, -b : size - b]
    mask = x ** 2 + y ** 2 <= r ** 2
    array = np.zeros((size, size))
    array[mask] = 1
    return array


def get_biggest_structure(bin_frame):
    # Label objects in image and filter for size of airfoil
    label_objects, _ = ndi.label(bin_frame)
    sizes = np.bincount(label_objects.ravel())
    sizes[0] = 0  # Deleting background
    mask_sizes = sizes == np.max(sizes)
    return mask_sizes[label_objects].astype(film.get_frame(0).dtype)


def fit(array, deg, x_vals=None):
    """Polyfit of array with given degree. Can take an interval of x-values in which to calculate output.
    
    Parameters
    ----------
    array : ndarray
        Array in format [points:axis] - first x then y axis, to be used for calculation, 
    deg : int
        Degree of interpolation (i.e.: 3 == cubic interpolation)
    x_vals : array-like, optional
        Interval of x_values in which to work with, by default None where the min() and max() of the orig array is used
    
    Returns
    -------
    fitted_array: ndarray
        Array in format [points:axis] - first x then y
    """
    if x_vals is None:
        x_vals = np.linspace(min(array[:, 0]), max(array[:, 0]), len(array))
    reg = np.polyfit(array[:, 0], array[:, 1], deg)
    poly = np.poly1d(reg)
    return np.vstack((x_vals, poly(x_vals))).T


def calc_len(array):
    array = np.cumsum(np.linalg.norm(np.gradient(array, axis=0), axis=-1), axis=0)
    return array


def find_idx_nearest(array, value):
    """Find the index in a 1D-array closest to a given value.
    
    Parameters
    ----------
    array : array-like
        Array where to search in 
    value : float
        Value to search for closest match
    
    Returns
    -------
    idx: int
        Index of array with closest match to value
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def split_outline(outline):
    outline = outline[outline[:, 0].argsort()]  # Sort for x
    lower_side = np.zeros((0, 2))
    upper_side = np.zeros((0, 2))
    for k in range(outline[0, 0], outline[-1, 0]):
        x_equal_k = outline[(outline[:, 0] == k)]  # grab all entries where x=i
        mean_at_x = np.mean(x_equal_k, axis=0)

        if len(x_equal_k) == 1:  # if only one point
            continue

        lower_than_mean = x_equal_k[(x_equal_k[:, 1] < mean_at_x[1])]
        lower_than_mean = np.mean(lower_than_mean, axis=0)
        lower_side = np.vstack((lower_side, lower_than_mean))

        higher_than_mean = x_equal_k[(x_equal_k[:, 1] > mean_at_x[1])]
        higher_than_mean = np.mean(higher_than_mean, axis=0)
        upper_side = np.vstack((upper_side, higher_than_mean))
    return lower_side, upper_side


def pos_reader(path_to_file):
    """Reads airfoil position from Json file, returns angle in deg"""
    with open(path_to_file, "r") as f:
        raw = json.load(f)
    rawpos = np.array(raw[1])
    triggertime = np.argmin(raw[10])
    pos = rawpos[triggertime:]
    pos = np.interp(np.linspace(0, 1, len(pos) * 4), np.linspace(0, 1, len(pos)), pos)
    pos = pos[: film.image_count]
    return pos


def spline(array, x_new):
    """Creates a cubic spline of a given 2D-array and returns an array: [points:axis]. 
    Also takes x positions for starting and stopping and the number of points to return.

    Parameters
    ----------
    array : array-like
        2D-array which is used for the spline in the format [points:axis]
    start : int
        Start value of x
    stop : int
        Stop value of x
    x_points : int, optional
        Number of values the array should have, by default 200
    
    Returns
    -------
    spline_array: ndarray
        Array in format [points:axis]
    """
    spline_obj = interp1d(array[:, 0], array[:, 1], kind="quadratic")
    return np.vstack((x_new, spline_obj(x_new))).T


fn_film = "film.cine"
fn_angles = "pos_feedback.dat"
film = get_file()
pivot = np.array((545, 438))
chord_len = 720
c_pivot = 0.326
c_flexrigid = 0.25
pos_array = -pos_reader(fn_angles)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    frame = 200
    outline = compute_outline(frame, 5)
    midline_points = compute_midline(frame, outline)
    midline = fit(
        midline_points,
        4,
        np.linspace(midline_points[0, 0], midline_points[-1, 0], 500),
    )
    plt.axis("equal")
    background_image = denoise(get_frame(10064), 99)
    background_image = ndi.binary_dilation(background_image, np.ones((3, 3)))
    plt.imshow(background_image.T)
