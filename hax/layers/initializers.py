import jax
import jax.numpy as jnp


def normal_initializer_mean(stddev=1e-2, mean=0.0):
  """
  Creates a JAX normal initializer with a custom mean and standard deviation.

  Args:
    stddev: The standard deviation of the normal distribution.
    mean: The mean of the normal distribution.

  Returns:
    A JAX initializer function.
  """
  def init(key, shape, dtype=jnp.float_):
    return jax.random.normal(key, shape, dtype) * stddev + mean
  return init

def uniform(minval=-1, maxval=1):
  def init(key, shape, dtype=jnp.float_):
    return jax.random.uniform(key, shape, dtype, minval=minval, maxval=maxval)
  return init

