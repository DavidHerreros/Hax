import jax.numpy as jnp
from flax import nnx
from flax.nnx.nn import initializers
from flax.typing import Initializer, Dtype

default_kernel_init = initializers.glorot_uniform()
default_bias_init = initializers.zeros_init()


class HyperLinear(nnx.Module):
    """
    A hypernetwork-driven linear layer using Flax NNX.

    Attributes:
      in_features:   size of the input x’s last dimension
      out_features:  size of the output’s last dimension
      hyper_features: size of the conditioning vector
      num_hyper_layers: number of hyper-layers
      use_bias:      whether to generate and add a bias term
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        in_hyper_features: int,
        hidden_hyper_features: int,
        num_hyper_layers: int = 1,
        *,
        use_bias: bool = True,
        dtype: Dtype | None = None,
        kernel_init: Initializer = default_kernel_init,
        bias_init: Initializer = default_bias_init,
        rngs: nnx.Rngs
    ):
        # store dims
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias

        # Hypernetwork that maps cond → flattened params
        total_params = in_features * out_features + (out_features if use_bias else 0)
        self.hyper = []
        self.hyper.append(nnx.Linear(in_features=in_hyper_features, out_features=hidden_hyper_features, kernel_init=kernel_init,
                                     bias_init=bias_init, dtype=dtype, rngs=rngs))
        for i in range(num_hyper_layers - 1):
            self.hyper.append(nnx.Linear(in_features=hidden_hyper_features, out_features=hidden_hyper_features, kernel_init=kernel_init,
                                         bias_init=bias_init, dtype=dtype, rngs=rngs))
        self.hyper.append(nnx.Linear(in_features=hidden_hyper_features, out_features=total_params, kernel_init=kernel_init,
                                     bias_init=bias_init, dtype=dtype, rngs=rngs))

    def __call__(self, x: jnp.ndarray, cond: jnp.ndarray) -> jnp.ndarray:
        # generate all params
        for layer in self.hyper[:-1]:
            cond = nnx.relu(layer(cond))
        cond = self.hyper[-1](cond)

        # split weight / bias
        w_flat = cond[..., : self.in_features * self.out_features]
        W = w_flat.reshape(*cond.shape[:-1], self.in_features, self.out_features)

        if self.use_bias:
            b = cond[..., self.in_features * self.out_features :]
        else:
            b = None

        # apply: y = x @ W + b
        y = jnp.einsum("...i,...io->...o", x, W)
        if self.use_bias:
            y = y + b
        return y