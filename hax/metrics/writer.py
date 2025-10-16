#!/usr/bin/env python


import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from skimage import measure
from scipy.stats import median_abs_deviation
from scipy.ndimage import gaussian_filter

from xmipp_metadata.image_handler import ImageHandler

from torch.utils.tensorboard import SummaryWriter

import jax
from jax import device_get
from jax.numpy import ndarray as JaxArray

from hax.utils import bcolors


class JaxSummaryWriter(SummaryWriter):
    def __init__(self, log_dir=None, **kwargs):
        super(JaxSummaryWriter, self).__init__(log_dir=os.path.join(log_dir, datetime.now().strftime("%Y%m%d-%H%M%S")), **kwargs)

    def _to_numpy(self, x):
        # If it’s a JAX array, pull it to host and convert
        if isinstance(x, JaxArray):
            return np.asarray(device_get(x))
        return x

    def _convert_tree(self, tree):
        # Recursively map over lists/tuples/dicts
        return jax.tree_util.tree_map(self._to_numpy, tree)

    def add_volumes(self, volumes, mode="structural"):
        volumes_ori = self._to_numpy(volumes)

        # 1) Resize volumes (size of 64 px for space and performance purposes)
        volumes = []
        for volume in volumes_ori:
            volumes.append(ImageHandler().scaleSplines(inputFn=volume, finalDimension=64))
        volumes = np.asarray(volumes)

        # 2) Compute MAD and mean volume
        volumes_mad = []
        volumes_series = []
        for vol in volumes:
            volumes_series.append(ImageHandler().generateMask(inputFn=vol, boxsize=64))
            volumes_mad.append(vol * volumes_series[-1])
        volume_mad = median_abs_deviation(volumes_mad, axis=0)
        volume_mean = volumes.mean(axis=0)
        volumes_series = np.asarray(volumes_series)

        # 3) Create automatic mask and mask original volume
        volume_mean_mask = ImageHandler().generateMask(inputFn=volume_mean, boxsize=64)
        volume_mean = volume_mean * volume_mean_mask
        volume_mad = volume_mad * volume_mean_mask
        volume_mean = gaussian_filter(volume_mean, sigma=1.0)
        volume_mad = gaussian_filter(volume_mad, sigma=1.0)

        # 4) Create triangulation
        verts_mean, faces_mean, _, _ = measure.marching_cubes(volume_mean,
                                                              level=0.8 * 0.5 * (volume_mean.max() + volume_mean.min()),
                                                              spacing=(1, 1, 1))
        verts_masks_b, faces_masks_b, color_masks_b = [], [], []
        for volume in volumes_series:
            if mode == "superpose":
                # VERSION: Place two meshes in the same tile
                verts_masks, faces_masks, _, _ = measure.marching_cubes(volume, level=0.001, spacing=(1, 1, 1))
                colors_masks = np.tile(np.asarray((194., 0., 61.))[None, ...], (verts_masks.shape[0], 1))
                colors_other = np.tile(np.asarray((61., 0., 194.))[None, ...], (verts_masks.shape[0], 1))
                verts_masks_b.append(np.concatenate([verts_masks, verts_mean], axis=0))
                faces_masks_b.append(np.concatenate([faces_masks, verts_masks.shape[0] + faces_mean], axis=0))
                color_masks_b.append(np.concatenate([colors_masks, colors_other], axis=0))
            elif mode == "structural":
                # VERSION: Structural differences based on masks
                volume = 2. * volume + volume_mean_mask
                verts_masks, faces_masks, _, volume_values = measure.marching_cubes(volume, level=0.001, spacing=(1, 1, 1))
                color_masks = np.zeros((verts_masks.shape[0], 3))
                color_masks[volume_values == 1] = np.array((61., 0., 194.))
                color_masks[volume_values == 2] = np.array((194., 0., 61.))
                color_masks[volume_values == 3] = np.array((255., 255., 255.))
                verts_masks_b.append(verts_masks)
                faces_masks_b.append(faces_masks)
                color_masks_b.append(color_masks)
            else:
                raise ValueError(f"Not valid value for mode. Valid values are "
                                 f"{bcolors.UNDERLINE}{bcolors.BOLD}superpose{bcolors.ENDC} or "
                                 f"{bcolors.UNDERLINE}{bcolors.BOLD}structural{bcolors.ENDC}")

        # 5) Get colors for MAD volume
        values_mad = volume_mad[verts_mean[..., 0].astype(int), verts_mean[..., 1].astype(int), verts_mean[..., 2].astype(int)]  # TODO: Check this (indexing is wrong and creates holes)
        cmap = plt.get_cmap('bwr')
        norm = plt.Normalize(vmin=values_mad.min(), vmax=values_mad.max())
        normalized_values = norm(values_mad)
        rgba_colors = cmap(normalized_values) * 255.
        colors_mad = rgba_colors[:, :3]

        # 6) Prepare vertices
        verts_mean = ((verts_mean - 0.5 * volume_mean.shape[-1]) / volume_mean.shape[-1])
        verts_mean = np.stack([verts_mean[..., 2], verts_mean[..., 1], verts_mean[..., 0]], axis=-1)
        for idx in range(volumes_series.shape[0]):
            verts_masks_b[idx] = ((verts_masks_b[idx] - 0.5 * volume_mean.shape[-1]) / volume_mean.shape[-1])
            verts_masks_b[idx] = np.stack([verts_masks_b[idx][..., 2], verts_masks_b[idx][..., 1], verts_masks_b[idx][..., 0]], axis=-1)

        # 7) Batch dimension for TensorBoard Mesh: [B, N, 3] and [B, M, 3]
        verts_mean_b = verts_mean[np.newaxis, ...].astype(np.float32)  # [1, V, 3]
        faces_mean_b = faces_mean[np.newaxis, ...].astype(np.int32)  # [1, F, 3]
        colors_mad_b = colors_mad[np.newaxis, ...].astype(np.float32)  # [1, V, 3]
        for idx in range(volumes_series.shape[0]):
            verts_masks_b[idx] = verts_masks_b[idx][np.newaxis, ...].astype(np.float32)  # [N, 1, V, 3]
            faces_masks_b[idx] = faces_masks_b[idx][np.newaxis, ...].astype(np.int32)  # [N, 1, F, 3]
            color_masks_b[idx] = color_masks_b[idx][np.newaxis, ...].astype(np.float32)  # [N, 1, V, 3]

        # 8) Create config fict to improve visualization of the mesh
        config_dict = {
            "camera": {
                "cls": "PerspectiveCamera",
                "fov": 90,
                "aspect": 0.5,
            },
            "lights": [
                # soft ambient fill
                {"cls": "AmbientLight", "color": [1.0, 1.0, 1.0], "intensity": 0.6},

                # key light from top-right
                {"cls": "DirectionalLight", "color": [1.0, 1.0, 1.0],
                 "intensity": 0.8, "position": [0.8, 0.8, 1.0]},

                # secondary light from lower-left
                {"cls": "DirectionalLight", "color": [1.0, 1.0, 1.0],
                 "intensity": 0.5, "position": [-0.7, -0.6, 0.8]},

                # subtle backlight (rim light)
                {"cls": "DirectionalLight", "color": [1.0, 1.0, 1.0],
                 "intensity": 0.9, "position": [0.0, 0.0, -1.0]}
            ],
            "material": {
                "cls": "MeshPhongMaterial",
                "vertexColors": False,  # use vertex grayscale or map color
                "wireframe": False,
                "shininess": 10,  # lower = matte (ChimeraX smooth)
                "specular": [0.1, 0.1, 0.1],  # soft highlights
                "transparent": True,
                "opacity": 1.0,  # slightly translucent for realism
            },
        }

        # 9) Write volumes to Tensorboard
        self.add_mesh(
            "Consensus volume",
            vertices=verts_mean_b,
            faces=faces_mean_b,
            config_dict=config_dict,
            global_step=0
        )

        self.add_mesh(
            "Variation volume",
            vertices=verts_mean_b,
            faces=faces_mean_b,
            colors=colors_mad_b,
            config_dict=config_dict,
            global_step=0
        )

        for idx in range(volumes_series.shape[0]):
            self.add_mesh(
                "Volume series",
                vertices=verts_masks_b[idx],
                faces=faces_masks_b[idx],
                colors=color_masks_b[idx],
                config_dict=config_dict,
                global_step=idx
            )


    def __getattribute__(self, name):
        # 1) Always let internal/private names through unwrapped:
        if name.startswith("_"):
            return object.__getattribute__(self, name)

        # 2) If the base class doesn’t define this attr, pass it straight through:
        if not hasattr(SummaryWriter, name):
            return object.__getattribute__(self, name)

        # 3) Grab the real method from *self* (so bound correctly):
        target = object.__getattribute__(self, name)

        # 4) If it isn’t callable, return it as-is:
        if not callable(target):
            return target

        # 5) Otherwise return our wrapper:
        def wrapped(*args, **kwargs):
            # cheap check: do we actually need to convert?
            leaves = jax.tree_util.tree_leaves((args, kwargs))
            if not any(isinstance(x, JaxArray) for x in leaves):
                return target(*args, **kwargs)

            # only pay conversion cost when we see a JAX array
            args_np = self._convert_tree(args)
            kwargs_np = self._convert_tree(kwargs)
            return target(*args_np, **kwargs_np)

        return wrapped




def main():
    import argparse
    import time
    from tensorboard import program
    from hax.utils import bcolors

    parser = argparse.ArgumentParser(
        description="Launch TensorBoard for a given JAX SummaryWriter log directory"
    )
    parser.add_argument(
        "--logdir", type=str, required=True,
        help="Path to the TensorBoard log directory"
    )
    args = parser.parse_args()

    # Launch TensorBoard programmatically
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', args.logdir])
    url = tb.launch()
    print(f"{bcolors.WARNING}TensorBoard is running at {url}")
    print(f"Press {bcolors.UNDERLINE}{bcolors.BOLD}Ctrl+C{bcolors.ENDC} {bcolors.WARNING}to stop.{bcolors.ENDC}")

    # Block until Ctrl-C
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nReceived Ctrl+C—shutting down.")

if __name__ == "__main__":
    main()
