import jax.numpy as jnp
from jax import lax, random, vmap
from jax.lax import scan
from jax.debug import print as jprint


def run_sde(rng, sde, ts, initial_sample, y=None, noise_last_step=False):
    if y is not None:
        drift = lambda t, x: sde.drift(t, x, y)
    else:
        drift = sde.drift
    
    zeros = jnp.zeros_like(initial_sample, dtype=jnp.float32)

    def step(carry, params):
        t, dt, is_last = params
        x, rng = carry

        rng, srng = random.split(rng)
        # ts = jnp.ones(x.shape[0]) * t

        noise = random.normal(srng, x.shape, dtype=jnp.float32)

        # dBt_orig = jnp.sqrt(dt) * sigma(t, x) * noise
        if not noise_last_step:
            noise = lax.cond(is_last, lambda _noise: zeros, lambda _noise: _noise, noise)

        dBt = noise
        drift_eval = drift(t, x)
        sigma_dBt = sde.sigma(t, x, dBt)
        # jprint("t_shape: {t}, x_shape: {x}", t=t.shape, x=x.shape)
        # jprint("t:{t}, x_norm_em: {x}", x=jnp.linalg.norm(x), t=t)
        # jprint("t: {t}, x: {x}, drift_eval: {drift_eval}, dBt: {dBt}, sigma_dBt: {sigma_dBt}", t=t, x=jnp.any(jnp.isnan(x)), drift_eval=jnp.any(jnp.isnan(drift_eval)), dBt=jnp.any(jnp.isnan(dBt)), sigma_dBt=jnp.any(jnp.isnan(sigma_dBt)))
        x = x + dt * drift_eval + dt**0.5 * sigma_dBt
        # jprint("t: {t}, x after: {x}", x=jnp.any(jnp.isnan(x)), t=t)
        return (x, rng), (x, drift_eval, dBt)

    dts = jnp.abs(ts[1:] - ts[:-1])
    # is_last = jnp.zeros_like(dts).at[-1].set(1)
    is_last = jnp.full(dts.shape[0], False, dtype=bool)
    is_last = is_last.at[-1].set(True)

    params = jnp.stack([ts[:-1], dts, is_last], axis=1)

    (x, (xs, drift_evals, dBts)) = scan(step, (initial_sample, rng), params)
    # breakpoint_if_nonfinite(xs)

    # xs = jnp.insert(xs, 0, initial_sample)
    xs = jnp.concatenate([initial_sample[None, :], xs])

    return xs, drift_evals, dBts

