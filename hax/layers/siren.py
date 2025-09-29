from jax import random as jnr
from jax._src.nn.initializers import _compute_fans
from jax import numpy as jnp


def siren_init(omega=1.0, c=1.0, in_axis=-2, out_axis=-1, dtype=jnp.float32):
    def init(key, shape, dtype=dtype):
        # shape = core.as_named_shape(shape)
        fan_in, fan_out = _compute_fans(shape, in_axis, out_axis)
        variance = jnp.sqrt(c / (3. * fan_in)) / omega
        return jnr.uniform(key, shape, dtype, minval=-variance, maxval=variance)

    return init


def siren_init_first(c=1.0, in_axis=-2, out_axis=-1, dtype=jnp.float32):
    def init(key, shape, dtype=dtype):
        # shape = core.as_named_shape(shape)
        fan_in, fan_out = _compute_fans(shape, in_axis, out_axis)
        variance = c * (1. / fan_in)
        return jnr.uniform(key, shape, dtype, minval=-variance, maxval=variance)

    return init


def bias_uniform(in_axis=-2, out_axis=-1, dtype=jnp.float32):
    # this is what Pytorch default Linear uses.
    def init(key, shape, dtype=dtype):
        # shape = core.as_named_shape(shape)
        fan_in, fan_out = _compute_fans(shape, in_axis, out_axis)
        variance = jnp.sqrt(1 / fan_in)
        return jnr.uniform(
            key, (int(fan_out),), dtype, minval=-variance, maxval=variance
        )

    return init
