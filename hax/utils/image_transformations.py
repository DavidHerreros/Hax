import jax
import jax.numpy as jnp


def apply_batch_translations(images, translations):
    """
    Applies (B, 2) translations to (B, M, M, 1) images.

    Args:
        images: Array of shape (B, M, M, 1)
        translations: Array of shape (B, 2) -> [dy, dx]
    """
    B, M, _, C = images.shape

    # Define the operation for a single image
    def shift_single(img, translation):
        return jax.image.scale_and_translate(
            image=img,
            shape=img.shape,  # Output shape (M, M, 1)
            spatial_dims=(0, 1),  # Only scale/translate the M x M dims
            scale=jnp.ones(2),  # No scaling (1.0 for all dims)
            translation=translation,
            method='linear'  # Can also use 'cubic' or 'lanczos3'
        )

    # Vectorize over the batch dimension
    translations = jnp.stack([translations[..., 1], translations[..., 0]], axis=-1)  # [dy, dx]
    return jax.vmap(shift_single)(images, translations)
