"""
A JAX-based utility for whitening images, particularly for low-SNR data
like Cryo-EM where clean noise patches are unavailable.

The intended workflow is a two-step process:

1.  **ONE-TIME SETUP:** Use `estimate_noise_psd` on a large, representative
    batch of images from your dataset. This computes the canonical noise
    profile (a 1D Power Spectral Density). Save this 1D array to disk.

2.  **ROUTINE PRE-PROCESSING:** In your data pipeline, load the saved noise PSD.
    Use `create_whitening_fn` to generate a fast, JIT-compiled whitening
    function. Use this function to pre-process all your image batches.
"""

import jax
import jax.numpy as jnp
import jax.numpy.fft as fft
from jax import vmap, jit
from typing import Callable, Tuple


@jit
def _compute_1d_psd_from_image(image: jax.Array) -> jax.Array:
    """
    Internal helper to compute the 1D radially averaged PSD from a single 2D image.
    """
    # 1. Compute 2D PSD
    shifted_fft = fft.fftshift(fft.fft2(image))
    psd_2d = jnp.abs(shifted_fft) ** 2

    # 2. Compute 1D Radial Average
    height, width = psd_2d.shape
    center_y, center_x = height // 2, width // 2

    y, x = jnp.indices(psd_2d.shape)
    r = jnp.sqrt((x - center_x) ** 2 + (y - center_y) ** 2).astype(jnp.int32)

    # max_radius = (jnp.sqrt(center_x ** 2 + center_y ** 2) + 1).astype(jnp.int32)
    max_radius = int(1.414 * center_x)

    total_power = jnp.bincount(r.ravel(), weights=psd_2d.ravel(), length=max_radius)
    pixel_count = jnp.bincount(r.ravel(), length=max_radius)

    radial_psd = total_power / jnp.maximum(pixel_count, 1)

    return radial_psd


def estimate_noise_psd(
        representative_batch: jax.Array,
        percentile: int = 10
) -> jax.Array:
    """
    Estimates the noise PSD from a batch of signal-plus-noise images.

    This function should be run once on a large, representative batch of data
    to establish the dataset's canonical noise profile.

    Args:
        representative_batch: A large batch of images, shape (B, H, W, 1) or (B, H, W).
        percentile: The percentile to use for robust estimation of the noise floor.

    Returns:
        The estimated 1D noise PSD array.
    """
    if representative_batch.ndim not in [3, 4]:
        raise ValueError("Input batch must have 3 or 4 dimensions.")

    # Squeeze channel dimension if it exists
    if representative_batch.ndim == 4:
        image_batch = jnp.squeeze(representative_batch, axis=-1)
    else:
        image_batch = representative_batch

    # vmap the helper function to efficiently compute the PSD for every image
    batch_of_psds = vmap(_compute_1d_psd_from_image)(image_batch)

    # Compute a low percentile across the batch to get a robust noise floor estimate
    noise_psd_estimate = jnp.percentile(batch_of_psds, q=percentile, axis=0)

    return noise_psd_estimate


def create_whitening_fn(
        noise_psd_1d: jax.Array,
        image_shape: Tuple[int, int]
) -> Callable[[jax.Array], jax.Array]:
    """
    Creates and JIT-compiles a function to whiten batches of images.

    This factory pre-builds the whitening filter from the dataset's noise profile
    for maximum efficiency.

    Args:
        noise_psd_1d: The pre-computed 1D noise PSD of the dataset from `estimate_noise_psd`.
        image_shape: The (height, width) of the images to be processed.

    Returns:
        A fast, JIT-compiled function that takes an image batch (B, H, W, 1) and
        returns the whitened batch.
    """
    height, width = image_shape

    # Pre-compute the 2D whitening filter from the 1D PSD
    center_y, center_x = height // 2, width // 2
    y, x = jnp.indices(image_shape)
    r = jnp.sqrt((x - center_x) ** 2 + (y - center_y) ** 2).astype(jnp.int32)

    # The filter is the inverse of the square root of the power spectrum
    # Add a small epsilon to prevent division by zero
    radial_filter = 1.0 / (jnp.sqrt(noise_psd_1d) + 1e-8)

    # Map the 1D filter values back to a 2D grid and unshift for multiplication
    whitening_filter_2d_shifted = radial_filter[r]
    whitening_filter_2d = fft.ifftshift(whitening_filter_2d_shifted)

    # This is the final function that will be returned
    @jit
    def whiten_batch(image_batch: jax.Array) -> jax.Array:
        """
        Applies the pre-computed whitening filter to a batch of images.
        Expected input shape: (B, H, W, 1).
        """
        if image_batch.ndim != 4 or image_batch.shape[1:3] != image_shape:
            raise ValueError(f"Input batch must have shape (B, {height}, {width}, 1)")

        images_squeezed = jnp.squeeze(image_batch, axis=-1)

        # Apply the filter in Fourier space.
        # JAX handles broadcasting the (H, W) filter across the (B, H, W) batch.
        fft_images = fft.fft2(images_squeezed)
        whitened_fft = fft_images * whitening_filter_2d

        whitened_images = jnp.real(fft.ifft2(whitened_fft))

        return jnp.expand_dims(whitened_images, axis=-1)

    return whiten_batch
