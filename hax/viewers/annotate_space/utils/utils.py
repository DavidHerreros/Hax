

import os
import subprocess
import numpy as np
from PIL import Image

from PyQt5.QtWidgets import QFileDialog

import hax.viewers.annotate_space.viewer_socket.server as server


def getImagePath(image_name):
    """Returns the absolute path to an image within the package."""
    module_dir = os.path.dirname(__file__)  # Get the directory of this module
    image_path = os.path.join(module_dir, '..', 'media', image_name)
    return image_path

def getServerProgram(env_name=None, variables=None):
    """Build the command to call the server script."""
    program = "python " + server.__file__

    if variables is None:
        variables = ""
    else:
        variables = ' '.join(f"{key}={value}" for key, value in variables.items())

    program = f"{getCondaActivationCommand()} && conda activate {env_name} && {variables} {program}"

    return program

def getCondaBase():
    try:
        conda_base = subprocess.check_output("conda info --base", shell=True, text=True).strip()
        return conda_base
    except subprocess.CalledProcessError as e:
        print(f"Error finding Conda base: {e}")
        return None

def getCondaActivationCommand():
    return f'eval "$({getCondaBase()}/bin/conda shell.bash hook)"'

def save_viewer_screenshot_with_dpi(
    viewer,
    dpi=300,
    *,
    width_in=None,
    height_in=None,
    canvas_only=True,
    flash=False,
    transparent_bg=False,   # <— NEW
    bg_tolerance=50          # <— color closeness threshold for transparency
):
    """
    Take a napari screenshot and save it with a custom DPI.
    Optionally enforce print size (inches) and make the canvas background transparent.

    transparent_bg: if True, pixels matching the canvas background become fully transparent.
    bg_tolerance: 0..255 (per-channel absolute difference tolerated for background match).
    """
    # --- compute target pixel size from desired print inches (optional) ---
    size = None
    if width_in is not None or height_in is not None:
        qv = viewer.window._qt_viewer
        cw = qv.canvas.size().width() or 1
        ch = qv.canvas.size().height() or 1
        aspect = ch / cw
        if width_in is not None and height_in is None:
            target_w = int(round(width_in * dpi))
            target_h = int(round(target_w * aspect))
        elif height_in is not None and width_in is None:
            target_h = int(round(height_in * dpi))
            target_w = int(round(target_h / aspect))
        else:
            target_w = int(round(width_in * dpi))
            target_h = int(round(height_in * dpi))
        size = (max(1, target_w), max(1, target_h))

    # --- screenshot as NumPy array (H, W, 3 or 4) ---
    arr = viewer.screenshot(size=size, canvas_only=canvas_only, flash=flash)

    # --- optional transparency: knock out the canvas background color ---
    if transparent_bg:
        # guess the canvas background color from the image borders
        # (napari canvas background is uniform; sampling the four corners is robust)
        h, w = arr.shape[:2]
        corners = np.array([arr[0, 0], arr[0, -1], arr[-1, 0], arr[-1, -1]], dtype=np.int16)
        bg_col = np.median(corners, axis=0).astype(np.uint8)  # robust against antialiasing

        # ensure RGBA
        if arr.shape[2] == 3:
            rgba = np.concatenate([arr, 255 * np.ones((h, w, 1), dtype=np.uint8)], axis=2)
        else:
            rgba = arr.copy()

        # build a mask where pixel ≈ background (per-channel tolerance)
        diff = np.abs(rgba[:, :, :3].astype(np.int16) - bg_col[:3].astype(np.int16))
        mask = (diff <= bg_tolerance).all(axis=2)

        # zero alpha on background-like pixels
        rgba[mask, 3] = 0
        arr = rgba

    # --- file dialog ---
    parent = getattr(viewer.window, "_qt_window", None)
    filters = "PNG (*.png);;JPEG (*.jpg *.jpeg);;TIFF (*.tif *.tiff)"
    path, selected_filter = QFileDialog.getSaveFileName(parent, "Save screenshot", "", filters)
    if not path:
        return None

    root, ext = os.path.splitext(path)
    ext = (ext or ".png").lower()
    path = root + ext
    if transparent_bg and ext in (".jpg", ".jpeg"):
        # JPEG has no alpha; switch to PNG automatically
        path = root + ".png"
        ext = ".png"

    # --- save with DPI (and alpha if present) ---
    im = Image.fromarray(arr)
    save_kwargs = {}
    if ext in (".jpg", ".jpeg"):
        save_kwargs["quality"] = 95
    im.save(path, dpi=(dpi, dpi), **save_kwargs)
    return path