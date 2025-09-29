from flax import nnx

# Define a default initializer function
default_init = nnx.initializers.glorot_uniform()


class Linear(nnx.Linear):
  """A Linear layer that defaults to glorot_uniform initialization."""
  def __init__(self, *args, kernel_init=default_init, **kwargs):
    # Call the parent's __init__, passing all arguments along.
    super().__init__(*args, kernel_init=kernel_init, **kwargs)


class Conv(nnx.Conv):
  """A Conv layer that defaults to glorot_uniform initialization."""
  def __init__(self, *args, kernel_init=default_init, **kwargs):
    # Call the parent's __init__, passing all arguments along.
    super().__init__(*args, kernel_init=kernel_init, **kwargs)


class ConvTranspose(nnx.ConvTranspose):
  """A ConvTranspose layer that defaults to glorot_uniform initialization."""
  def __init__(self, *args, kernel_init=default_init, **kwargs):
    # Call the parent's __init__, passing all arguments along.
    super().__init__(*args, kernel_init=kernel_init, **kwargs)
