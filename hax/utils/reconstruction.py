

import sys
from functools import partial
from tqdm import tqdm
import jax.numpy as jnp
from jax import random, vmap, jit
from jax.scipy.ndimage import map_coordinates


# ---------- Helpers ----------

def _rand_rot(key):
    """Random SO(3) via QR; ensures det=+1."""
    A = random.normal(key, (3, 3))
    Q, R = jnp.linalg.qr(A)
    # Make det(Q)=+1
    sign = jnp.sign(jnp.linalg.det(Q))
    Q = Q.at[:, 2].set(Q[:, 2] * sign)
    return Q  # 3x3

def _make_grid(nz, ny, nx):
    """World grid coordinates in [-1, 1]^3, shaped (3, nz, ny, nx)."""
    z = jnp.linspace(-1.0, 1.0, nz)
    y = jnp.linspace(-1.0, 1.0, ny)
    x = jnp.linspace(-1.0, 1.0, nx)
    Z, Y, X = jnp.meshgrid(z, y, x, indexing="ij")
    return jnp.stack([X, Y, Z], axis=0)  # (3, Z, Y, X)

def _world_to_stack_coords(R, grid, nz, ny, nx, ih, iw):
    """
    Map world coords to stack coords for sampling a (D,H,W) stack built from image.
    stack axes order: (depth, v, u) = (z, y, x)
    """
    # world -> camera/stack space: s = R^T * w
    X, Y, Z = grid[0], grid[1], grid[2]  # (Z,Y,X)
    w = jnp.stack([X, Y, Z], axis=0)     # (3, Z,Y,X)
    s = jnp.einsum("ij,jxyz->ixyz", R.T, w)                       # (3, Z,Y,X)

    xs, ys, zs = s[0], s[1], s[2]  # in [-1,1]

    # Scale from [-1,1] to index space [0..N-1]
    def to_idx(a, N):
        return (a + 1.0) * 0.5 * (N - 1.0)

    u = to_idx(xs, iw)  # width
    v = to_idx(ys, ih)  # height
    d = to_idx(zs, nz)  # depth
    return d, v, u  # each (Z,Y,X)

@partial(jit, static_argnums=[3,])
def _accumulate_one_image(image, R, grid, vol_shape):
    """
    Back-project one image into the volume by repeating it across depth,
    rotating coords back into the stack, sampling, and accumulating.
    """
    nz, ny, nx = vol_shape
    ih, iw = image.shape

    # Build the stack (depth, v, u) by repeating the image along depth
    # shape: (nz, ih, iw)
    stack = jnp.broadcast_to(image, (nz, ih, iw))

    # Map world-grid coordinates to this stack's index coordinates
    d, v, u = _world_to_stack_coords(R, grid, nz, ny, nx, ih, iw)

    # Sample stack at (d, v, u) with trilinear interpolation
    # map_coordinates expects coords shaped (ndim, ...)
    coords = jnp.stack([d, v, u], axis=0)  # (3, Z,Y,X)
    sampled = map_coordinates(stack, coords, order=1, mode="constant", cval=0.0)

    # Also accumulate a weight mask for simple normalization
    in_bounds = (
        (d >= 0) & (d <= nz - 1) &
        (v >= 0) & (v <= ih - 1) &
        (u >= 0) & (u <= iw - 1)
    )
    weight = in_bounds.astype(sampled.dtype)
    return sampled, weight  # (Z,Y,X), (Z,Y,X)

# Vectorize over a batch of images/rotations
_batch_accumulate = vmap(_accumulate_one_image, in_axes=(0, 0, None, None))

# ---------- Public API ----------

# @partial(jit, static_argnames=["vol_shape", "data_loader"])
def reconstruct_volume_streaming(key, vol_shape, data_loader):
    """
    Args:
      batches: iterable (Python) of JAX arrays shaped (B, H, W).
               You can pass e.g. a list/tuple of DeviceArrays.
      key: jax.random.PRNGKey.
      vol_shape: (Z, Y, X) tuple for the output volume grid.

    Returns:
      volume: (Z, Y, X) reconstruction from all batches.
      rotations_per_batch: list of arrays, each (B, 3, 3) rotations used.
    """
    nz, ny, nx = vol_shape
    grid = _make_grid(nz, ny, nx)

    vol = jnp.zeros(vol_shape, dtype=jnp.float32)
    wsum = jnp.zeros(vol_shape, dtype=jnp.float32)

    rotations_per_batch = []

    # Note: Python loop is fine — inner math is JIT/Vmapped.
    pbar = tqdm(data_loader, desc=f"Progress", file=sys.stdout, ascii=" >=", colour="green")
    for (batch, _) in pbar:
        # batch: (B, H, W)
        B = batch.shape[0]

        # random rotations for this batch
        subkeys = random.split(key, B + 1)
        key = subkeys[0]
        Rs = vmap(_rand_rot)(subkeys[1:])  # (B, 3, 3)
        rotations_per_batch.append(Rs)

        # accumulate for the whole batch
        sampled_b, weight_b = _batch_accumulate(batch[..., 0], Rs, grid, vol_shape)  # (B, Z,Y,X) each

        # sum over batch and add
        vol = vol + sampled_b.sum(axis=0)
        wsum = wsum + weight_b.sum(axis=0)

    # avoid divide-by-zero
    volume = jnp.where(wsum > 0, vol / jnp.clip(wsum, a_min=1e-6), 0.0)
    return volume, rotations_per_batch