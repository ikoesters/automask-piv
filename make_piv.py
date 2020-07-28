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

import datetime
import os
from math import log
from pathlib import Path

import fluidimage
from fluidimage.topologies.piv import TopologyPIV

os.environ["OMP_NUM_THREADS"] = "1"


def piv_from_png(save_path, image_folder):

    correl_val = 0.3
    overlap_val = 0.5
    window_size_val = 128

    path_src = save_path + image_folder

    save_path = (
        save_path
        + f"piv_correl{correl_val}_overlap{overlap_val}_window_size{window_size_val}"
    )

    if os.path.isdir(save_path) is False:
        os.mkdir(save_path)
    os.chmod(save_path, 0o774)

    postfix = datetime.datetime.now().isoformat()

    params = TopologyPIV.create_default_params()

    params.series.path = path_src
    params.series.ind_start = 1
    params.series.ind_step = 1
    params.series.strcouple = "i:i+2"

    params.piv0.shape_crop_im0 = window_size_val
    params.piv0.grid.overlap = overlap_val
    params.multipass.number = int(log(window_size_val, 2) - 4)  # last window is 32x32
    params.multipass.use_tps = "True"
    params.fix.correl_min = correl_val

    params.saving.how = "recompute"
    params.saving.path = save_path
    params.saving.postfix = postfix

    topology = TopologyPIV(params)
    topology.compute(sequential=False)

    paths = Path(save_path).glob("*.h5")
    for path in paths:
        os.chmod(path, 0o774)


if __name__ == "__main__":
    pass
