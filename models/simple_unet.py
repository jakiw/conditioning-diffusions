import flax.linen as nn
import jax.numpy as jnp
import optax


class ApproximateScore(nn.Module):
    """A simple model with multiple fully connected layers and some fourier features for the time variable."""

    T = 1.0
    # control_until : float
    # bb_residual : bool
    layers: list[int] = (256, 128, 64, 32)

    @nn.compact
    def __call__(self, t, x, y):

        # T = self.T
        # control_until = self.control_until

        # t = t[:, None]
        t_orig = t

        in_size = x.shape[0]

        z = jnp.concatenate([jnp.array([t]), x, y])
        act = nn.relu
        skips = []
        for layer in self.layers:
            z = nn.Dense(layer)(z)
            z = act(z)
            skips.append(z)

        for i in range(len(self.layers) - 1):
            layer = self.layers[-1 - i]
            skip = skips[-1 - i]
            z = nn.Dense(layer)(z)
            z = jnp.concatenate([z, skip])
            z = act(z)

        z = nn.Dense(in_size)(z)

        # if self.bb_residual:
        #     bb_term = (y-x)/jnp.sqrt(T-t_orig)
        # else:
        #     bb_term = 0

        return z  # + bb_term * (1 - control_active)
        # return y1
