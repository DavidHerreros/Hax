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
