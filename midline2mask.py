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

import os

import numpy as np
from scipy.signal import butter, filtfilt
from skimage.draw import polygon

import frame2midline as f2m


def filter_midlines(midlines, order=2, fs=1000, cutoff=100):
    midlines = np.copy(midlines)
    for mid_point in range(midlines.shape[1]):
        for coo in range(midlines.shape[2]):
            midlines[:, mid_point, coo] = butter_lowpass_filter(
                midlines[:, mid_point, coo], cutoff, fs, order
            )
    return midlines


def make_naca(midline_points, apply_corr_val):
    # Generate midline from points
    midline = f2m.fit(
        midline_points,
        4,
        np.linspace(midline_points[0, 0], midline_points[-1, 0], 500),
    )

    # Make coordinate system for each point on midline
    # Make distance vectors of length==1
    midline_norm = np.linalg.norm(np.gradient(midline, axis=0), axis=1)
    midline_coo = np.gradient(midline, axis=0) / np.repeat(
        midline_norm[:, np.newaxis], 2, axis=-1
    )
    # Rotate by 90 degrees counterclockwise
    midline_coo = np.fliplr(midline_coo)
    midline_coo[:, 0] *= -1

    # Generate Naca foil
    naca = sym_naca_foil(0.18, midline.shape[0]) * f2m.chord_len
    naca[:, 1] = np.flip(naca[:, 1])

    # Put foil outline around midline
    naca_vector = np.repeat(naca[:, 1, np.newaxis], 2, axis=-1)
    corr_val = np.linspace(30, 5, midline.shape[0])
    if apply_corr_val == True:
        corr_val = np.append(
            corr_val, np.zeros(midline.shape[0] - corr_val.shape[0])
        )  # increase slope of corr_val and stay at 0 => above: len(midline) //2
    else:
        corr_val = np.linspace(0, 0, len(midline))

    corr_val = np.repeat(corr_val[:, np.newaxis], 2, axis=-1)
    naca_upper = midline + midline_coo * (naca_vector + corr_val)
    naca_lower = midline - midline_coo * (naca_vector + corr_val)
    return naca_upper, naca_lower


def generate_mask(midline):
    naca_upper, naca_lower = make_naca(midline, apply_corr_val=True)
    mask = indices_to_maskarray(naca_upper, naca_lower)
    mask = decrop(mask, f2m.get_frame(0))
    return mask


def indices_to_maskarray(naca_upper, naca_lower):
    im = np.ones_like(f2m.crop(f2m.get_frame(0)))
    perimiter = np.vstack((naca_lower, np.flipud(naca_upper)))
    rr, cc = polygon(perimiter[:, 0], perimiter[:, 1])
    im[rr, cc] = 0
    return im


def sym_naca_foil(t, sample_rate):
    """Generate 2D data of a scaled symmetric 4-digit-NACA-foil scaled in coord
    t is thickness in per cent to cord
    cord is cord length"""
    x = np.linspace(0, 1, sample_rate)
    np.meshgrid(x)
    curve = (
        5
        * t
        * (
            0.2969 * np.sqrt(x)
            - 0.126 * x
            - 0.3516 * x ** 2
            + 0.2843 * x ** 3
            - 0.1015 * x ** 4
        )
    )
    naca = np.vstack((x, curve))
    return naca.T


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def interpolate_midlines(midlines, factor):
    interp_midlines = np.zeros(
        (midlines.shape[0] * factor, midlines.shape[1], midlines.shape[2])
    )
    for mid_point in range(midlines.shape[1]):
        for coo in range(midlines.shape[2]):
            interp_midlines[:, mid_point, coo] = np.interp(
                np.linspace(0, 1, midlines.shape[0] * factor,),
                np.linspace(0, 1, midlines.shape[0],),
                midlines[:, mid_point, coo],
            )
    return interp_midlines


def decrop(mask_array, orig_frame, xstart=300, xend=1150, ystart=0, yend=800):
    full_mask = np.ones_like(orig_frame)
    full_mask[xstart:xend, ystart:yend] = mask_array
    return full_mask


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    midlines = np.load("midlines_testfile.npy")

    midlines_filt = filter_midlines(midlines)
    midlines_inter = interpolate_midlines(midlines_filt, 4)
    # plt.plot(midlines[:, -2, 1])
    # plt.plot(filt_midlines[:, 0, 1])
