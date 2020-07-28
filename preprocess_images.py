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
from skimage.filters import threshold_local

from frame2midline import get_frame


def get_denoised_frame_localthresh(frame_nb):
    frame = f2m.get_frame(frame_nb)
    frame = local_thresh(frame, 15, 15)
    frame = normalize(frame)
    return frame


def local_thresh(frame, block_size, offset):
    local_thresholds = threshold_local(frame, block_size, "gaussian", offset)
    thresh_f = frame - local_thresholds
    thresh_f = np.where(thresh_f < 0, 0, thresh_f)
    return thresh_f


def normalize(frame):
    frame = (frame - np.min(frame)) / (np.max(frame - np.min(frame)))
    return frame


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    raw_frame = preprocess_frame(0)
    plt.imshow(raw_frame.T)
