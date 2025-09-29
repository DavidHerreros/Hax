import jax
import jax.numpy as jnp
import numpy as np


def gather_nd_jax(params, indices, batch_dim=1):
    """ A JAX/NumPy porting of gather_nd implementation.
    This implementation can handle leading batch dimensions in params.

    Args:
        params: A JAX array of dimension [b1, ..., bn, g1, ..., gm, c].
        indices: A JAX array of dimension [b1, ..., bn, x, m].
                'm' must be equal to the number of grid dimensions (len([g1, ..., gm])).
        batch_dim: Integer, indicates how many leading dimensions are batch dimensions (n in the example).

    Returns:
        gathered: A JAX array of dimension [b1, ..., bn, x, c].

    Example:
    >>> key = jax.random.PRNGKey(0)
    >>> key_params, key_pos = jax.random.split(key)
    >>> batch_shape = (2, 3) # b1, b2 (n=2)
    >>> grid_shape = (4, 5, 6) # g1, g2, g3 (m=3)
    >>> channels = 7 # c
    >>> num_gather_points = 10 # x
    >>> params_shape = batch_shape + grid_shape + (channels,)
    >>> indices_shape = batch_shape + (num_gather_points, len(grid_shape))
    >>> params_np = np.random.randn(*params_shape).astype(np.float32)
    >>> # Ensure indices are within bounds [0, grid_dim_size - 1]
    >>> indices_np_list = []
    >>> for i, dim_size in enumerate(grid_shape):
    ...     indices_np_list.append(np.random.randint(0, dim_size, size=batch_shape + (num_gather_points, 1)))
    >>> indices_np = np.concatenate(indices_np_list, axis=-1).astype(np.int32)
    >>> params_jax = jnp.array(params_np)
    >>> indices_jax = jnp.array(indices_np)
    >>> gathered_jax = gather_nd_jax(params_jax, indices_jax, batch_dim=len(batch_shape))
    >>> # Expected output shape: batch_shape + (num_gather_points, channels)
    >>> print(f"Gathered JAX shape: {gathered_jax.shape}")
    Gathered JAX shape: (2, 3, 10, 7)
    """
    if not isinstance(params, jnp.ndarray) or not isinstance(indices, jnp.ndarray):
        raise TypeError("Input 'params' and 'indices' must be JAX arrays.")

    original_params_shape = params.shape
    original_indices_shape = indices.shape

    if batch_dim < 0:
        raise ValueError("batch_dim must be non-negative.")
    if batch_dim > len(original_params_shape) - 1:  # -1 because of channel dim
        raise ValueError("batch_dim exceeds params rank (excluding channel dimension).")
    if batch_dim > len(original_indices_shape) - 2:  # -2 because of x and m dims
        raise ValueError("batch_dim exceeds indices rank (excluding last two dimensions).")

    original_batch_dims_shape_tuple = original_params_shape[:batch_dim]

    if batch_dim == 0:
        effective_batch_size = 1
    else:
        effective_batch_size = int(np.prod(original_params_shape[:batch_dim]))

    c_dim = original_params_shape[-1]
    grid_dims_tuple = original_params_shape[batch_dim:-1]
    num_grid_dims = len(grid_dims_tuple)  # m

    num_indices_to_gather = original_indices_shape[-2]  # x
    indices_coord_dims = original_indices_shape[-1]  # m'

    if num_grid_dims != indices_coord_dims:
        raise ValueError(
            f"The last dimension of indices ({indices_coord_dims}) must be equal to the "
            f"number of grid dimensions in params ({num_grid_dims})."
        )

    # Reshape leading batch dims to a single batch dim
    # Target shape for params: [B, g1, ..., gm, c] where B = product of original batch dims
    params_reshaped = params.reshape((effective_batch_size,) + grid_dims_tuple + (c_dim,))
    # Target shape for indices: [B, x, m]
    indices_reshaped = indices.reshape((effective_batch_size, num_indices_to_gather, num_grid_dims))

    # Build gather indices for JAX advanced indexing
    # We want: gathered[b, i, :] = params_reshaped[b, indices_reshaped[b, i, 0], ..., indices_reshaped[b, i, m-1], :]

    # Batch indices for the flattened batch dimension of params_reshaped
    # Shape: (effective_batch_size, 1)
    batch_idx_component = jnp.arange(effective_batch_size)[:, None]

    # Components for indexing grid dimensions
    # Each component will have shape (effective_batch_size, num_indices_to_gather)
    grid_idx_components = [indices_reshaped[..., i] for i in range(num_grid_dims)]

    # Full indexing tuple for JAX.
    # JAX will broadcast batch_idx_component (B, 1) with grid_idx_components (B, X)
    # to effectively index at (B, X) for each grid dimension.
    indexing_tuple = (batch_idx_component,) + tuple(grid_idx_components)

    gathered = params_reshaped[indexing_tuple]
    # Shape of gathered will be (effective_batch_size, num_indices_to_gather, c_dim)

    # Reshape back to include original batch dimensions
    # Target shape: [b1, ..., bn, x, c]
    final_shape = original_batch_dims_shape_tuple + (num_indices_to_gather, c_dim)
    gathered_final = gathered.reshape(final_shape)

    return gathered_final


def interpolate(grid_3d, sampling_points):
    """Trilinear interpolation on a 3D regular grid using JAX.

    Args:
        grid_3d: A JAX array with shape `[A1, ..., An, H, W, D, C]` where H, W, D are
                 height, width, depth of the grid and C is the number of channels.
        sampling_points: A JAX array with shape `[A1, ..., An, M, 3]` where M is the
                         number of sampling points. Sampling points are in the coordinate
                         system of the grid (e.g., if grid is HxWxD, points are in [0,H-1]x[0,W-1]x[0,D-1]).
                         Points outside the grid are projected to the grid borders by clipping.

    Returns:
        A JAX array of shape `[A1, ..., An, M, C]`
    """
    if not isinstance(grid_3d, jnp.ndarray) or not isinstance(sampling_points, jnp.ndarray):
        raise TypeError("Input 'grid_3d' and 'sampling_points' must be JAX arrays.")

    grid_3d_shape = grid_3d.shape
    sampling_points_shape = sampling_points.shape

    # Voxel cube dimensions [H, W, D]
    voxel_cube_dims = jnp.array(grid_3d_shape[-4:-1], dtype=jnp.int32)
    # Batch dimensions [A1, ..., An]
    # These are the dimensions before H, W, D, C in grid_3d
    # and before M, 3 in sampling_points
    batch_dims_shape_tuple = sampling_points_shape[:-2]
    num_batch_dims = len(batch_dims_shape_tuple)
    num_points = sampling_points_shape[-2]  # M
    num_channels = grid_3d_shape[-1]  # C

    # Calculate coordinates of the 8 surrounding voxels
    bottom_left_float = jnp.floor(sampling_points)
    top_right_float = bottom_left_float + 1.0

    # Cast to integer for indexing
    bottom_left_idx = bottom_left_float.astype(jnp.int32)
    top_right_idx = top_right_float.astype(jnp.int32)

    # Unbind coordinates for easier manipulation
    x0_idx, y0_idx, z0_idx = [bottom_left_idx[..., i] for i in range(3)]
    x1_idx, y1_idx, z1_idx = [top_right_idx[..., i] for i in range(3)]

    # Create indices for the 8 corner points of the surrounding cube for each sampling point
    # Shape of each index_coord will be [*batch_dims_shape_tuple, num_points * 8]
    # (Indices for C000, C100, C010, C110, C001, C101, C011, C111)
    # The order here determines how weights are later applied.
    # Standard trilinear order:
    # (x0,y0,z0), (x1,y0,z0), (x0,y1,z0), (x1,y1,z0)
    # (x0,y0,z1), (x1,y0,z1), (x0,y1,z1), (x1,y1,z1)

    corners_x_idx = jnp.concatenate([x0_idx, x1_idx, x0_idx, x1_idx,
                                     x0_idx, x1_idx, x0_idx, x1_idx], axis=-1)
    corners_y_idx = jnp.concatenate([y0_idx, y0_idx, y1_idx, y1_idx,
                                     y0_idx, y0_idx, y1_idx, y1_idx], axis=-1)
    corners_z_idx = jnp.concatenate([z0_idx, z0_idx, z0_idx, z0_idx,
                                     z1_idx, z1_idx, z1_idx, z1_idx], axis=-1)

    # Stack to form [..., num_points*8, 3] indices
    indices_for_gather = jnp.stack([corners_x_idx, corners_y_idx, corners_z_idx], axis=-1)

    # Clip indices to be within grid boundaries [0, H-1], [0, W-1], [0, D-1]
    # max_indices is (H-1, W-1, D-1)
    max_indices = voxel_cube_dims - 1
    indices_for_gather_clipped = jnp.clip(indices_for_gather, 0, max_indices)  # Broadcasting handles this

    # Gather content from the grid at the 8 corner points
    # batch_dim for gather_nd is the number of batch dimensions in grid_3d and indices_for_gather_clipped
    # grid_3d has shape [A1..An, H, W, D, C]. batch_dim for gather is num_batch_dims.
    # indices_for_gather_clipped has shape [A1..An, M*8, 3]. batch_dim for gather is num_batch_dims.
    corner_contents = gather_nd_jax(
        params=grid_3d,
        indices=indices_for_gather_clipped,
        batch_dim=num_batch_dims
    )
    # corner_contents shape: [*batch_dims_shape_tuple, num_points*8, C]

    # Calculate weights for trilinear interpolation
    # Distances from sampling point to bottom-left corner (normalized coordinates within the unit cube)
    dist_to_bottom_left = sampling_points - bottom_left_float  # dx, dy, dz
    # Distances from sampling point to top-right corner (1 - (dx,dy,dz))
    dist_to_top_right = top_right_float - sampling_points  # 1-dx, 1-dy, 1-dz

    dx, dy, dz = [dist_to_bottom_left[..., i] for i in range(3)]
    one_minus_dx, one_minus_dy, one_minus_dz = [dist_to_top_right[..., i] for i in range(3)]

    # Weights for the 8 corners, matching the order of indices_for_gather
    # w000 = (1-dx)(1-dy)(1-dz)
    # w100 = dx(1-dy)(1-dz)
    # w010 = (1-dx)dy(1-dz)
    # ... and so on.
    weights_x = jnp.concatenate([one_minus_dx, dx, one_minus_dx, dx,
                                 one_minus_dx, dx, one_minus_dx, dx], axis=-1)
    weights_y = jnp.concatenate([one_minus_dy, one_minus_dy, dy, dy,
                                 one_minus_dy, one_minus_dy, dy, dy], axis=-1)
    weights_z = jnp.concatenate([one_minus_dz, one_minus_dz, one_minus_dz, one_minus_dz,
                                 dz, dz, dz, dz], axis=-1)

    weights = weights_x * weights_y * weights_z
    # weights shape: [*batch_dims_shape_tuple, num_points*8]
    weights = weights[..., None]  # Add channel dim for broadcasting: [..., num_points*8, 1]

    # Weighted sum of corner contents
    interpolated_values = weights * corner_contents
    # Shape: [*batch_dims_shape_tuple, num_points*8, C]

    # Reshape to separate the 8 corner contributions and sum them up
    # Reshape from [..., M*8, C] to [..., M, 8, C]
    final_shape_before_sum = batch_dims_shape_tuple + (num_points, 8, num_channels)
    interpolated_values_reshaped = interpolated_values.reshape(final_shape_before_sum)

    # Sum over the 8 corner contributions (axis with size 8)
    final_interpolated_values = jnp.sum(interpolated_values_reshaped, axis=-2)
    # final_interpolated_values shape: [*batch_dims_shape_tuple, num_points, C]

    return final_interpolated_values