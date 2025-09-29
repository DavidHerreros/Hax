import jax.numpy as jnp

def _rot(axis, theta):
    axis = jnp.asarray(axis, dtype=float)
    axis = axis / jnp.linalg.norm(axis)
    x, y, z = axis
    c = jnp.cos(theta)
    s = jnp.sin(theta)
    C = 1. - c
    return jnp.array([
        [c + x * x * C,     x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s,   c + y * y * C,   y * z * C - x * s],
        [z * x * C - y * s,   z * y * C + x * s, c + z * z * C  ],
    ])

def symmetry_matrices(sym):
    s = sym.strip().lower()

    if s.startswith('c') and s[1:].isdigit():
        n = int(s[1:])
        return jnp.array([_rot([0,0,1], 2. * jnp.pi * k / n) for k in range(n)])

    if s.startswith('d') and s[1:].isdigit():
        n = int(s[1:])
        mats = [_rot([0,0,1], 2. * jnp.pi * k / n) for k in range(n)]
        for k in range(n):
            phi = jnp.pi * k / n
            axis = [jnp.cos(phi), jnp.sin(phi), 0.0]
            mats.append(_rot(axis, jnp.pi))
        return jnp.array(mats)

    raise ValueError(f"Unsupported or unknown symmetry label: {sym}")
