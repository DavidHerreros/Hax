import numpy as np
from jax import numpy as jnp


def euler_matrix_batch(alpha, beta, gamma):

    ca = jnp.cos(alpha * (np.pi / 180.0))[:, None]
    sa = jnp.sin(alpha * (np.pi / 180.0))[:, None]
    cb = jnp.cos(beta * (np.pi / 180.0))[:, None]
    sb = jnp.sin(beta * (np.pi / 180.0))[:, None]
    cg = jnp.cos(gamma * (np.pi / 180.0))[:, None]
    sg = jnp.sin(gamma * (np.pi / 180.0))[:, None]

    cc = cb * ca
    cs = cb * sa
    sc = sb * ca
    ss = sb * sa

    row_1 = jnp.concat([cg * cc - sg * sa, cg * cs + sg * ca, -cg * sb], axis=1)

    row_2 = jnp.concat([-sg * cc - cg * sa, -sg * cs + cg * ca, sg * sb], axis=1)

    row_3 = jnp.concat([sc, ss, cb], axis=1)

    return jnp.stack([row_1, row_2, row_3], axis=1)


def euler_from_matrix(matrix):
    # Only valid for Xmipp axes szyz
    firstaxis, parity, repetition, frame = (2, 1, 1, 0)
    _EPS = jnp.finfo(float).eps * 4.0
    _NEXT_AXIS = [1, 2, 0, 1] # axis sequences for Euler angles

    i = firstaxis
    j = _NEXT_AXIS[i + parity]
    k = _NEXT_AXIS[i - parity + 1]

    M = jnp.array(matrix, dtype=jnp.float32, copy=False)
    if repetition:
        sy = jnp.sqrt(M[i, j] * M[i, j] + M[i, k] * M[i, k])
        ax = jnp.where(sy > _EPS, jnp.atan2(M[i, j], M[i, k]), jnp.atan2(-M[j, k], M[j, j]))
        ay = jnp.atan2(sy, M[i, i])
        az = jnp.where(sy > _EPS, jnp.atan2(M[j, i], -M[k, i]), 0.0)
    else:
        cy = jnp.sqrt(M[i, i] * M[i, i] + M[j, i] * M[j, i])
        ax = jnp.where(cy > _EPS, jnp.atan2(M[k, j], M[k, k]), jnp.atan2(-M[j, k], M[j, j]))
        ay = jnp.atan2(-M[k, i], cy)
        az = jnp.where(cy > _EPS, jnp.atan2(M[j, i], M[i, i]), 0.0)

    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return jnp.stack([ax, ay, az], axis=0)
