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


import numpy as np

from PyQt5.QtCore import QThread, pyqtSignal

from sklearn.cluster import MiniBatchKMeans


class ClusteringQThread(QThread):
    finished = pyqtSignal()
    centers_labels = pyqtSignal(list)

    def __init__(self, n_clusters, z_space, mode, axis=None):
        super().__init__()
        self.n_clusters = n_clusters
        self.data = z_space
        self.mode = mode
        self.axis = axis

    def run(self):
        if self.mode == "KMeans":
            # Compute KMeans
            clusters = MiniBatchKMeans(n_clusters=self.n_clusters).fit(self.data)
            centers, labels = clusters.cluster_centers_, clusters.labels_

        elif self.mode == "Along_Dim":
            # Determine the range of PCA DIM and divide into X equal intervals
            pca_axis = self.data[..., self.axis]
            min_pca1, max_pca1 = pca_axis.min(), pca_axis.max()
            intervals = np.linspace(min_pca1, max_pca1, self.n_clusters + 1)
            means_pca1 = 0.5 * (intervals[:-1] + intervals[1:])

            # Initialize lists to hold group means and point indices
            group_means = np.zeros((self.n_clusters, self.data.shape[-1]))
            labels = np.empty_like(pca_axis)
            # Compute clusters along dimension and save automatic selection
            for i in range(self.n_clusters):
                # Find points that fall within the current interval
                in_interval = (pca_axis >= intervals[i]) & (pca_axis < intervals[i + 1])
                if i == self.n_clusters - 1:
                    # Ensure the last group includes the max value
                    in_interval = (pca_axis >= intervals[i]) & (pca_axis <= intervals[i + 1])
                points_in_group = self.data[in_interval, self.axis]

                if len(points_in_group) > 0:
                    # Assign labels
                    labels[in_interval] = i

                # Store the mean and the points
                group_means[i, self.axis] = means_pca1[i]

            centers, labels = group_means, labels
        else:
            raise ValueError("Clustering method not implemented")

        # Emit results
        self.centers_labels.emit([centers, labels])
        self.finished.emit()
