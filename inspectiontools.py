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

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
import seaborn as sns
import skimage.filters as filt
from imageio import imwrite
from matplotlib import rc
from scipy import stats
from tqdm import trange

import frame2midline as f2m
import midline2mask as m2m
from main import make_folder
from main import sum_amount


# sns.set_style("whitegrid")
sns.set_style("white")


def mean_displacement_of_tracer(array):
    """
    array in shape: (points, cartesian_coordinates) eg. (4,2)
    """
    grad = np.gradient(array, axis=0)
    disp = np.linalg.norm(grad, axis=-1)
    return np.mean(disp)


def pos_reader(path_to_file):
    """Reads airfoil position from Json file, returns angle in deg"""
    with open(path_to_file, "r") as f:
        raw = json.load(f)
    rawpos = np.array(raw[1])
    triggertime = np.argmin(raw[10])
    pos = rawpos[triggertime:]
    pos = np.interp(np.linspace(0, 1, len(pos) * 4), np.linspace(0, 1, len(pos)), pos)
    pos = pos[: f2m.film.image_count]
    return pos


def count_structures(bin_frame):
    # Label objects in image and filter for size of airfoil
    label_objects, _ = ndi.label(bin_frame)
    sizes = np.bincount(label_objects.ravel())
    sizes[0] = 0  # Deleting background
    return label_objects, sizes


def segment_reflections(bin_frame):
    # Label objects in image and filter for size of airfoil
    label_objects, _ = ndi.label(bin_frame)
    sizes = np.bincount(label_objects.ravel())
    sizes[0] = 0  # Deleting background
    mask_sizes = sizes > 20
    return mask_sizes[label_objects]


def mask_reflections(bin_frame):
    # Label objects in image and filter for size of airfoil
    label_objects, _ = ndi.label(bin_frame)
    sizes = np.bincount(label_objects.ravel())
    mask_sizes = sizes < 10
    return mask_sizes[label_objects], sizes[1:]


def denoise_sobel(frame, threshold):
    sobel_f = ndi.sobel(frame / ndi.median(frame))
    thresh_f = np.where(sobel_f > threshold, 1, 0)
    return thresh_f


def find_pivot(frame_nb, pivot):
    a = f2m.sum_frames(frame_nb, 8)
    pos_array = f2m.pos_array
    pos = pos_array[frame_nb]
    chord_len = 720
    lastpoint = np.array(
        (
            np.cos(np.deg2rad(pos)) * 0.326 * chord_len + pivot[0],
            np.sin(np.deg2rad(pos)) * 0.326 * chord_len + pivot[1],
        )
    )
    plt.imshow(a.T)
    plt.scatter(pivot[0], pivot[1])
    plt.plot(np.linspace(pivot[0], lastpoint[0]), np.linspace(pivot[1], lastpoint[1]))


def vel_reader(path_to_file):
    with open(path_to_file, "r") as pos_file:
        data = json.load(pos_file)
    return data[8]


def get_period_length(pos_feedback):
    """finds the zero crossings and gets the period length of a given
    periodic function"""

    counts = []
    pos_old = 0
    idx_old = 0
    zero_count = 0
    for idx, pos in enumerate(pos_feedback):
        if pos_old < 0 and pos >= 0:
            zero_count += 1
        if zero_count is 1:
            counts.append(idx - idx_old)
            zero_count = 0
            idx_old = idx
        pos_old = pos
    T = np.mean(counts[1:-1])
    freq = 1000 / T
    period_beginnings = np.cumsum(counts)
    return T, freq, period_beginnings


def plot_and_save_image(image, name, im_format):
    plt.imshow(image.T)
    plt.savefig(f"./plots/{name}.{im_format}")


def plot_and_save(*args, **kwargs):
    plt.figure()
    if "im" in kwargs:
        plt.imshow(kwargs["im"].T)
    if args:
        for a in args:
            lineplot(a)
    if "xlabel" in kwargs:
        plt.xlabel(kwargs["xlabel"])
    if "ylabel" in kwargs:
        plt.ylabel(kwargs["ylabel"])
    plt.legend()
    plt.savefig(f"./plots/{kwargs['name']}.{kwargs['format']}")


def lineplot(array, x=0):
    """Takes array in [points:axis] format; x before y-axis. 
    If more than 2 dimensions exist, like with a relative position axis, the x index can be specified, y must follow.
    
    Parameters
    ----------
    array : ndarray
        Array which to plot, format [points:axis]
    x : int, optional
        Index of x-axis - y has to follow, by default 0
    """
    plt.plot(array[:, x], array[:, x + 1])
    return


def plot_image(array, dirname, filename, frame_nb, dpi):
    make_folder(dirname)
    plt.ioff()
    plt.style.use("seaborn-whitegrid")
    plt.close()
    fig = plt.figure()
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    array = np.flipud(array).T
    array = np.where(array == 1, 0, 1)
    frame_nb = frame_nb
    plt.imshow(array, aspect="equal")
    plt.savefig(dirname + filename + str(frame_nb).zfill(4) + ".png", dpi=dpi)
    plt.close()


if __name__ == "__main__":
    ## Noise Histogram
    #a = f2m.get_frame(1)
    #a_noise = np.where(a>120,0,a)
    #sns.distplot(a_noise.flatten(), kde=False, bins = 80)
    #plt.xlim((26,110))
    #plt.title("Noise Distribution in Dataset")
    #plt.xlabel("Pixel Value")
    #plt.ylabel("Number of Occurences")

    ## CCDs Noise Analysis
    #a1, a2,a3,a4 = a_noise[0:639,0:399], a_noise[640:1280,0:399], a_noise[0:639, 400:800], a_noise[640:1280, 400:800
    #sns.distplot(a1.flatten(), kde=False, bins = 80, label='a1')
    #sns.distplot(a4.flatten(), kde=False, bins = 80, label='a2')
    #plt.xlim((26,110))
    #plt.legend()

    ## Noise Start of Film vs End
    #b = f2m.get_frame(16500)
    #b_noise = np.where(b>120,0,b)
    #sns.distplot(a_noise.flatten(), bins=80, label='a', kde=False)
    #sns.distplot(b_noise.flatten(), bins=80, label='b', kde=False)
    #plt.legend()
    pass
