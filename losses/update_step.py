from functools import partial

import jax
import jax.numpy as jnp
import optax
from jax import jit

# Takes a loss function and performs a jitted update


@partial(jit, static_argnames=["sde", "loss_function", "grad_clip", "nn_model", "optimizer"])
def update_step(
    rng, loss_function, sde, nn_model, nn_params, ts, initial_samples, opt_state, optimizer, y_obs, grad_clip=1.0
):
    val, grads = jax.value_and_grad(loss_function, argnums=(3))(
        rng, sde, nn_model, nn_params, ts, initial_samples, y_obs
    )
    grad_norm = jnp.sqrt(jax.tree_util.tree_reduce(lambda acc, g: acc + jnp.sum(g**2), grads, initializer=0.0))

    if grad_clip >= 0:
        # grad_norm = jnp.sqrt(sum([jnp.sum(jnp.square(x)) for x in jax.tree_leaves(grad)]))
        # grad_norm = jnp.sqrt(jnp.sum(jnp.square(grads.flatten())))
        clipped_grad = jax.tree_util.tree_map(lambda x: x * grad_clip / jnp.maximum(grad_norm, grad_clip), grads)
    else:
        clipped_grad = grads

    updates, opt_state = optimizer.update(clipped_grad, opt_state)
    nn_params = optax.apply_updates(nn_params, updates)

    return val, nn_params, opt_state, grad_norm
