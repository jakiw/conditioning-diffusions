# Based on model at
# https://colab.research.google.com/drive/1SeXMpILhkJPjXUaesvzEhc3Ke6Zl_zxJ?usp=sharing#scrollTo=21v75FhSkfCq
# Changes: Removed the normalisation by the standard deviation of the marginal probability distribution

from typing import Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp


class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""

    embed_dim: int
    scale: float = 30.0

    @nn.compact
    def __call__(self, x):
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        W = self.param("W", jax.nn.initializers.normal(stddev=self.scale), (self.embed_dim // 2,))
        W = jax.lax.stop_gradient(W)
        x_proj = x[:, None] * W[None, :] * 2 * jnp.pi
        return jnp.concatenate([jnp.sin(x_proj), jnp.cos(x_proj)], axis=-1)


class Dense(nn.Module):
    """A fully connected layer that reshapes outputs to feature maps."""

    output_dim: int

    @nn.compact
    def __call__(self, x):
        return nn.Dense(self.output_dim)(x)[:, None, None, :]


class UNet(nn.Module):
    """A time-dependent score-based model built upon U-Net architecture.

    Args:
        channels: The number of channels for feature maps of each resolution.
        embed_dim: The dimensionality of Gaussian random feature embeddings.
    """

    channels: Tuple[int] = (32, 64, 128, 256)
    embed_dim: int = 256

    @nn.compact
    def __call__(self, x, t):
        # The swish activation function
        act = nn.swish
        # Obtain the Gaussian random feature embedding for t
        embed = act(nn.Dense(self.embed_dim)(GaussianFourierProjection(embed_dim=self.embed_dim)(t)))

        # Upsampling
        h1 = nn.Conv(self.channels[0], (3, 3), (1, 1), padding="VALID", use_bias=False)(x)
        # Incorporate information from t
        h1 += Dense(self.channels[0])(embed)
        # Group normalization
        h1 = nn.GroupNorm(4)(h1)
        h1 = act(h1)

        h2 = nn.Conv(self.channels[1], (3, 3), (2, 2), padding="VALID", use_bias=False)(h1)
        h2 += Dense(self.channels[1])(embed)
        h2 = nn.GroupNorm()(h2)
        h2 = act(h2)

        h3 = nn.Conv(self.channels[2], (3, 3), (2, 2), padding="VALID", use_bias=False)(h2)
        h3 += Dense(self.channels[2])(embed)
        h3 = nn.GroupNorm()(h3)
        h3 = act(h3)

        h4 = nn.Conv(self.channels[3], (3, 3), (2, 2), padding="VALID", use_bias=False)(h3)
        h4 += Dense(self.channels[3])(embed)
        h4 = nn.GroupNorm()(h4)
        h4 = act(h4)

        # Downsampling
        h = nn.Conv(self.channels[2], (3, 3), (1, 1), padding=((2, 2), (2, 2)), input_dilation=(2, 2), use_bias=False)(
            h4
        )
        # Skip connection from the encoding path
        h += Dense(self.channels[2])(embed)
        h = nn.GroupNorm()(h)
        h = act(h)
        h = nn.Conv(self.channels[1], (3, 3), (1, 1), padding=((2, 3), (2, 3)), input_dilation=(2, 2), use_bias=False)(
            jnp.concatenate([h, h3], axis=-1)
        )
        h += Dense(self.channels[1])(embed)
        h = nn.GroupNorm()(h)
        h = act(h)
        h = nn.Conv(self.channels[0], (3, 3), (1, 1), padding=((2, 3), (2, 3)), input_dilation=(2, 2), use_bias=False)(
            jnp.concatenate([h, h2], axis=-1)
        )
        h += Dense(self.channels[0])(embed)
        h = nn.GroupNorm()(h)
        h = act(h)
        h = nn.Conv(1, (3, 3), (1, 1), padding=((2, 2), (2, 2)))(jnp.concatenate([h, h1], axis=-1))

        return h
