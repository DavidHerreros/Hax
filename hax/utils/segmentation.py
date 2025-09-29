import numpy as np
from scipy import ndimage as ndi

from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.morphology import ball


def watershed_segmentation(vol, mask, radius=10):
    distance = ndi.distance_transform_edt(vol * mask.astype(int))
    coords = peak_local_max(distance, footprint=ball(radius), labels=mask.astype(bool))
    mask_2 = np.zeros(distance.shape, dtype=bool)
    mask_2[tuple(coords.T)] = True
    markers, _ = ndi.label(mask_2)
    return watershed(-distance, markers, mask=mask.astype(bool))

def get_segmentation_centers(seg):
    indices = np.asarray(np.where(seg > 0)).T
    coords = np.stack([indices[:, 2], indices[:, 1], indices[:, 0]], axis=1)
    groups = seg[indices[:, 0], indices[:, 1], indices[:, 2]]

    centers = []
    for group in np.unique(groups):
        centers.append(np.mean(coords[groups == group], axis=0))
    centers = np.asarray(centers)

    return groups, centers