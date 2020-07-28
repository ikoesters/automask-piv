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
import subprocess
import time

import matplotlib.pyplot as plt
import numpy as np
from imageio import imwrite
from tqdm import tqdm, trange
from scipy import ndimage as ndi

import frame2midline as f2m
import make_piv
import midline2mask as m2m
import preprocess_images as pi
import inspectiontools as insp


def compute_midlines():
    len_midline_points = f2m.compute_midline(
        0, outline=f2m.compute_outline(0, sum_amount)
    ).shape[0]
    midlines = np.zeros(
        (int(np.ceil(how_many_ims / sum_amount)), len_midline_points, 2)
    )
    for idx, frame_nb in enumerate(
        tqdm(range(start_at_frame, start_at_frame + how_many_ims, sum_amount))
    ):
        outline = f2m.compute_outline(frame_nb, sum_amount)
        midlines[idx] = f2m.compute_midline(frame_nb, outline)
    return midlines


def im2im(image, frame_nb):
    return masked_image


def show_masks(midlines, start_at_frame):
    for idx, mid in enumerate(tqdm(midlines)):
        frame_nb = start_at_frame + idx
        masked_path = make_folder(save_path + image_folder)
        frame = f2m.get_frame(frame_nb)
        frame = f2m.denoise(f2m.crop(frame))
        frame = np.where(frame > 0, 1, 0)
        frame = ndi.binary_dilation(frame, np.ones((3, 3))).astype("uint")
        mask = m2m.generate_mask(mid)
        mask = f2m.crop(mask)
        mask = np.where(mask == 0, 1, 0)
        masked_frame = frame + mask
        masked_frame = np.where(masked_frame == 0, 0, 1)
        insp.plot_image(masked_frame, "./vis_mask/", "vismask", frame_nb, 100)
    return


def mask_ims(midlines, start_at_frame):
    for idx, mid in enumerate(tqdm(midlines)):
        frame_nb = start_at_frame + idx
        masked_path = make_folder(save_path + image_folder)
        frame = pi.get_denoised_frame_localthresh(frame_nb)
        mask = m2m.generate_mask(mid)
        masked_frame = frame * mask
        masked_frame = pi.normalize(masked_frame)
        masked_frame *= 2 ** 16 - 1
        imwrite(
            masked_path + "/masked_" + str(idx) + ".png",
            masked_frame.T.astype("uint16"),
            format="PNG",
        )
    return


def make_folder(path):
    """Creates folder if not already existent, assigns 774-rights (drwxrwxr--).
    
    Parameters
    ----------
    path : str
        Path to where the folder will be created
    """
    if not os.path.exists(path):
        os.makedirs(path)
    os.chmod(path, 0o774)
    return path


sum_amount = 5
how_many_ims = f2m.film.image_count - sum_amount  # f2m.film.image_count
start_at_frame = 0
save_path = "./PIV_1/"
image_folder = "masked/"

if __name__ == "__main__":
    make_folder(save_path)

    # Generating Midlines
    midlines = compute_midlines()
    np.save("midlines", midlines)

    midlines = m2m.filter_midlines(midlines)
    midlines = m2m.interpolate_midlines(midlines, sum_amount)
    np.save("midlines_filt_interp.npy", midlines)
    # midlines = midlines[:1000]

    # Mask images
    mask_ims(midlines, start_at_frame)

    # Compute PIV
    make_piv.piv_from_png(save_path, image_folder)
