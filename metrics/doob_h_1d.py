from functools import partial

import jax.numpy as jnp
from jax import jit, vmap
from jax.lax import scan


# Calclulate p(x_1 = y | x_t) for all t on a grid
@partial(jit, static_argnames=["sde"])
def get_pt_cond_y(sde, y, ts, xs):

    drift, sigma, a1, sigma_transp_inv = sde
    dx = xs[1] - xs[0]

    y_index = jnp.argmin(jnp.abs(xs - y))
    final = jnp.zeros_like(xs).at[y_index].set(1 / dx * 0.5)
    final = final.at[y_index - 1].set(1 / dx * 0.5)

    @partial(vmap, in_axes=(0, None, None, None))
    def back_at_value(x, t, dt, p_next):
        mean = x + dt * drift(t, x)
        var = dt * sigma(t, x, jnp.array([1.0])) ** 2
        transition_kernel = jnp.exp(-((xs - mean) ** 2) / (2 * var)) * 1 / jnp.sqrt(2 * jnp.pi * var)
        transition_kernel *= dx
        # dont want to normalize since otherwise TK at the borders where half of it is outside of
        # xs get double mass on the rest. therefore do actual lebesgue weights
        # ASSUMING A UNIFORM xs GRID!
        probability = jnp.sum(transition_kernel * p_next)
        return probability

    def step(carry, params):
        t, dt = params
        p_next = carry

        p_current = back_at_value(xs, t, dt, p_next)

        return (p_current), (p_current)

    # we go backwards, so pass in the ts backwards
    ts = ts[::-1]
    dts = jnp.abs(ts[1:] - ts[:-1])

    # if T = ts[-1] (of original ts value), wea assume we have final condition there
    # we will start at ts[-2] with dt = ts[-1] - ts[-2] to go the first step back and
    # end up at ts[0]
    (_, p_array) = scan(step, (final), (ts[1:], dts))

    p_array = p_array[::-1, :]
    return p_array


# take nable log of the above function to get doobs h transform
def get_doob_drift(sde, y, ts, xs):
    drift, sigma, a, sigma_transp_inv = sde
    p_array = get_pt_cond_y(sde, y, ts, xs)
    # dts = ts[1:] - ts[:-1]
    two_dxs = xs[2:] - xs[:-2]

    a1 = p_array[:, 2:]
    a2 = p_array[:, :-2]
    two_dxs = two_dxs
    # all of these work the same for me
    # log_diffs = jnp.log(a1/b)
    # log_diffs = (jnp.log(a1) - jnp.log(b))
    log_diffs = jnp.log(1 + (a1 - a2) / a2)
    log_derivatives = log_diffs / (two_dxs[None, :])

    log_derivatives = jnp.array(log_derivatives)

    xs_ = xs[1:-1]

    # probs are not defined at final time (diracte4533)
    ts_ = ts[:-1]

    def drift(t, x):
        # We round t down to the last value we have
        i_t = jnp.searchsorted(ts_, t, side="right") - 1
        i_x = jnp.argmin(jnp.abs(x - xs_))
        nabla_log = log_derivatives[i_t, i_x]
        nabla_log = a(t, x, nabla_log)
        return nabla_log

    return drift, log_derivatives
