import jax.numpy as jnp
from flax import nnx


class ResBlock(nnx.Module):
    def __init__(self, C: int, num_groups: int, dropout_prob: float, *, rngs: nnx.Rngs):
        self.relu = nnx.relu
        self.gnorm1 = nnx.GroupNorm(num_groups=num_groups, num_features=C, rngs=rngs, dtype=jnp.bfloat16)  # Esta capa normaliza los canales dividiendolas previamente en pequeños grupos
        self.gnorm2 = nnx.GroupNorm(num_groups=num_groups, num_features=C, rngs=rngs, dtype=jnp.bfloat16)  # Esta capa normaliza los canales dividiendolas previamente en pequeños grupos
        self.conv1 = nnx.Conv(C, C, kernel_size=(3, 3), strides=(1, 1), padding="SAME", rngs=rngs, dtype=jnp.bfloat16)
        self.conv2 = nnx.Conv(C, C, kernel_size=(3, 3), strides=(1, 1), padding="SAME", rngs=rngs, dtype=jnp.bfloat16)
        self.dropout = nnx.Dropout(dropout_prob, rngs=rngs)

    def __call__(self, x):
        r = self.conv1(self.relu(self.gnorm1(x)))
        r = self.dropout(r)
        r = self.conv2(self.relu(self.gnorm2(r)))
        return r + x