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


import cuml
cuml.accel.install()

import numpy as np

from PyQt5.QtCore import QThread, pyqtSignal

from umap import UMAP
from sklearn.decomposition import PCA


class DimRedQThread(QThread):
    finished = pyqtSignal()
    red_space = pyqtSignal(np.ndarray)

    def __init__(self, n_components, data, mode, params):
        super().__init__()
        self.n_components = n_components
        self.data = data
        self.mode = mode
        self.params = params

    def run(self):
        if self.mode == "PCA":
            # Compute PCA
            red_space = PCA(n_components=self.n_components, **self.params).fit_transform(self.data)

        elif self.mode == "UMAP":
            # Compute UMAP
            red_space = UMAP(n_components=self.n_components, **self.params).fit_transform(self.data)
        else:
            raise ValueError("Dimensionality reduction method not implemented")

        # Emit results
        self.red_space.emit(red_space)
        self.finished.emit()
