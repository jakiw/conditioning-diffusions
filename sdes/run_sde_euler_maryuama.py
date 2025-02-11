import jax.numpy as jnp
from jax import lax, random, vmap
from jax.lax import scan


def run_sde(rng, sde, ts, initial_sample, y=None, noise_last_step=False):

    if y is not None:
        drift = lambda t, x: sde.drift(t, x, y)
    else:
        drift = sde.drift

    zeros = jnp.zeros_like(initial_sample)

    def step(carry, params):
        t, dt, is_last = params
        x, rng = carry

        rng, srng = random.split(rng)
        # ts = jnp.ones(x.shape[0]) * t

        noise = random.normal(srng, x.shape)

        # dBt_orig = jnp.sqrt(dt) * sigma(t, x) * noise
        if not noise_last_step:
            noise = lax.cond(is_last, lambda _noise: zeros, lambda _noise: _noise, noise)

        dBt = jnp.sqrt(dt) * noise

        drift_eval = drift(t, x)
        x = x + dt * drift(t, x) + sde.sigma(t, x, dBt)

        return (x, rng), (x, drift_eval, dBt)

    dts = jnp.abs(ts[1:] - ts[:-1])
    # is_last = jnp.zeros_like(dts).at[-1].set(1)
    is_last = jnp.full(dts.shape[0], False, dtype=jnp.bool)
    is_last = is_last.at[-1].set(True)

    params = jnp.stack([ts[:-1], dts, is_last], axis=1)

    (x, (xs, drift_evals, dBts)) = scan(step, (initial_sample, rng), params)
    # breakpoint_if_nonfinite(xs)

    # xs = jnp.insert(xs, 0, initial_sample)
    xs = jnp.concatenate([initial_sample[None, :], xs])

    return xs, drift_evals, dBts

