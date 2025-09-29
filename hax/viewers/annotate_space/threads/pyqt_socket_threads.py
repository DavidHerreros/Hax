# **************************************************************************
# *
# * Authors:     David Herreros Calero (dherreros@cnb.csic.es)
# *
# * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# * GNU General Public License for more details.
# *
# * You should have received a copy of the GNU General Public License
# * along with this program; if not, write to the Free Software
# * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# * 02111-1307  USA
# *
# *  All comments concerning this program package may be sent to the
# *  e-mail address 'scipion@cnb.csic.es'
# *
# **************************************************************************


import sys
import os
import signal
import time
import subprocess

import numpy as np
from xmipp_metadata.image_handler import ImageHandler
from scipy.ndimage import gaussian_filter

from napari_molecule_reader.molecule_reader import read_molecules

from PyQt5.QtCore import QThread, pyqtSignal

from hax.viewers.annotate_space.viewer_socket.client import Client


def runProgram(program, args, env=None, cwd=None, popen=False):
    command = program + " " + args
    if popen:
        return subprocess.Popen(command, shell=True, stdout=sys.stdout, stderr=sys.stderr,
                                env=env, cwd=cwd, preexec_fn=os.setsid)
    else:
        return subprocess.check_call(command, shell=True, stdout=sys.stdout, stderr=sys.stderr,
                                     env=env, cwd=cwd)


class ServerQThread(QThread):
    def __init__(self, program, metadata_file, mode, port, env):
        super().__init__()
        self.program = program
        self.metadata_file = metadata_file
        self.mode = mode
        self.port = port
        self.env = env
        self.process = None

    def run(self):
        args = f"--metadata_file {self.metadata_file} --mode {self.mode} --port {self.port}"
        self.process = runProgram(self.program, args, popen=True)

    def stop(self):
        os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)

class ClientQThread(QThread):
    finished = pyqtSignal()
    volume = pyqtSignal(object)
    chimera = pyqtSignal()

    def __init__(self, port, path, mode):
        super().__init__()
        self.client = Client(port)
        self.path = path
        self.mode = mode
        self.z = None
        self.file_names = None
        self.vol_file_template = os.path.join(os.path.join(self.path, "decoded_map_class_{:02d}.mrc"))
        self.is_chimera_signal = False
        self.save_to_file = False

    def readMap(self, file):
        map = ImageHandler().read(file).getData()
        return np.squeeze(map)

    def readModel(self, file):
        # Read PDB or MMCIF file
        _, points = read_molecules(
            file)  # This returns coordinates and vectors in case we want to have a visualization based on bonds
        points = points[0][..., 1:]

        # Prepare grid to place coordinates
        grid = np.zeros((128, 128, 128))

        # Move center of mass to origin, scale, and displace to create the indices
        points = points - np.mean(points, axis=0)
        scale_factor = (0.5 * 128) / np.max(np.linalg.norm(points, axis=1))
        points = scale_factor * points + 0.5 * 128

        # Ensure coordinates are within grid bounds by rounding to the nearest integer
        rounded_coords = np.rint(points).astype(int)

        # Place the intensities into the grid
        intensities = np.ones(points.shape[0])
        for coord, intensity in zip(rounded_coords, intensities):
            x, y, z = coord
            # Check bounds to avoid out-of-range errors
            if 0 <= x < 128 and 0 <= y < 128 and 0 <= z < 128:
                grid[x, y, z] += intensity  # Accumulate intensity if multiple points map to the same voxel

        # Apply a Gaussian filter to smooth the grid
        sigma = 1.0  # Standard deviation for Gaussian filter
        smoothed_grid = gaussian_filter(grid, sigma=sigma)

        return smoothed_grid

    def run(self):
        if not self.mode == "FromFiles":
            np.savetxt(os.path.join(self.path, "z_server.txt"), self.z)
            while not os.path.isfile(os.path.join(self.path, "z_server.txt")) and not os.access(os.path.join(self.path, "z_server.txt"), os.R_OK):
                time.sleep(0.01)
            self.client.sendDataToSever(os.path.join(self.path, "z_server.txt"))
        else:
            with open(os.path.join(self.path, "z_server.pkl"), 'w') as f:
                for line in self.z:
                    f.write("%s\n" % line)
            self.client.sendDataToSever(os.path.join(self.path, "z_server.pkl"))

        # Emit signals
        if len(self.z) == 1:
            if self.vol_file_template.split(".")[-1] == "mrc":
                generated_map = self.readMap(self.vol_file_template.format(1))
            elif self.vol_file_template.split(".")[-1] == "pdb" or self.vol_file_template.split(".")[-1] == "cif":
                generated_map = self.readModel(self.vol_file_template.format(1))
            if self.save_to_file:
                ImageHandler().write(generated_map, self.vol_file_template.format(1), overwrite=True)
            else:
                self.volume.emit(generated_map)
                os.remove(self.vol_file_template.format(1))
        else:
            for idx in range(len(self.z)):
                new_path = os.path.join(self.path, self.file_names[idx] + ".mrc")
                formatted_vol_file = self.vol_file_template.format(idx + 1)
                if formatted_vol_file.split(".")[-1] == "pdb" or formatted_vol_file.split(".")[-1] == "cif":
                    generated_map = self.readModel(formatted_vol_file)
                    ImageHandler().write(generated_map, new_path)
                    os.remove(formatted_vol_file)
                elif formatted_vol_file.split(".")[-1] == "mrc" and formatted_vol_file != new_path:
                    ImageHandler().convert(formatted_vol_file, new_path, overwrite=True)
                    os.remove(formatted_vol_file)

        # Emit signals
        if len(self.z) > 1 or self.is_chimera_signal:
            self.chimera.emit()
            self.is_chimera_signal = False
        self.finished.emit()
