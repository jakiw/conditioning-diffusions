import jax.numpy as jnp
from jax import grad, jacfwd, jacobian, jacrev, jit, random, value_and_grad, vmap
from jax.lax import scan
from jax.debug import print as jprint


def solve_J_equation_2(rng, sde, ts, sample_path, sample_dBts, **kwargs):
    D = sample_path.shape[1]

    drift_jac = jacfwd(sde.drift, argnums=1)

    sigma_jac = jacfwd(sde.sigma, argnums=1)

    # integration_mode = "first_dBt"
    # if "integration_mode" in kwargs:
    #     assert kwargs["integration_mode"] in ["first_dBt", "first_J"]
    #     integration_mode = kwargs["integration_mode"]
    mode = "average"
    if "mode" in kwargs:
        # assert kwargs["mode"] in ["average", "last", "first"]
        mode = kwargs["mode"]

    def d_alpha_s_unnormalized(s):
        return (1 - s) ** (0.5 * (-1 + jnp.sqrt(5)))

    ts_reverse = ts[::-1]
    time_spent = ts[-1] - ts_reverse

    gamma = 0.5 * (-1 + jnp.sqrt(5))
    gamma = jnp.float32(gamma)
    alpha_factors = (time_spent[1:-1] / time_spent[2:]) ** gamma
    alpha_factors = jnp.concatenate([jnp.array([1.0]), alpha_factors], dtype=jnp.float32)
    # If ts are equidistant, this should result in the same thing as
    # a = jnp.arange(0, 10)
    # (a[1:-1]/a[2:])**gamma

    def step(carry, params):
        t, dt, x, dBt, is_first, alpha_factor = params
        doob, rng = carry

        rng, srng = random.split(rng)
        jprint("t: {t}, x_norm: {x}", t=t, x=jnp.linalg.norm(x))
        # SHOULD THERE BE NO DT IN FRONT OF SIGMA?
        propagated_doob = doob + dt * doob.T.dot(drift_jac(t, x)) + doob.T.dot(sigma_jac(t, x, dBt))

        # Basically two different ways to discretize \int J_{s | 0} sigma^{-T} (dBs)
        # Once approximating J_{s | 0} = J_{0 | 0} and once as the EM approx of J_{t | 0} = J_{0|0} + delta *
        sigma_dBt = sde.sigma_transp_inv(t, x, dBt)

        sigma_dBt_J_delta_t = sigma_dBt + dt * sigma_dBt.T.dot(drift_jac(t, x)) + sigma_dBt.T.dot(sigma_jac(t, x, dBt))

        # Basically the flow SDE just differentiates each part of the SDE with respect to x
        # Therefore we basically go back by d/dx (dt * drift) + d/dx (sigma * dBt)
        # Since doob is something that is applied to the derivative (not a x-value, but a direction)
        # it goes from the other side via the transpose
        if mode == "last":
            # dBt *= is_first
            # doob = dBt + J_increment.T.dot(doob)
            doob = (
                sigma_dBt * is_first + propagated_doob
            )  # dt * doob.T.dot(drift_jac(t, x)) + doob.T.dot(sigma_jac(t, x, dBt))
            # doob += sigma_dBt * is_first + dt * doob.T.dot(drift_jac(t, x)) + doob.T.dot(sigma_jac(t, x, dBt))
        elif mode == "average":
            doob = sigma_dBt + propagated_doob
            # doob += dBt + dt * doob.T.dot(drift_jac(t, x)) + dt * dBt.T.dot(drift_jac(t, x))
        elif mode == "first":
            # doob = dBt
            doob = sigma_dBt_J_delta_t
        elif mode == "optimal":
            doob = sigma_dBt + propagated_doob * alpha_factor
        else:
            raise Exception(f"Mode '{mode}' not supported")
        # elif mode == "soc":
        # THIS sets x_{N-1} to the bridge value instead of x_N !
        #     initial = (-1 - x) * is_first / dt
        #     doob += initial + dt * doob.T.dot(drift_jac(t, x))
        #     # jprint("Doob is {d}", d=doob)

        return (doob, rng), (doob)

    reverse_ts = ts[::-1]
    dts = jnp.abs(reverse_ts[1:] - reverse_ts[:-1])
    reverse_ts = reverse_ts[1:]

    reverse_paths = sample_path[::-1, :]
    reverse_paths = reverse_paths[1:, :]
    reverse_noise = sample_dBts[::-1, :]

    is_first = jnp.zeros(reverse_paths.shape[0])
    is_first = is_first.at[0].set(1)

    doob_initial = jnp.zeros(D, dtype=jnp.float32)

    # print shape of all the arrays inputted into params
    params = [reverse_ts, dts, reverse_paths, reverse_noise, is_first, alpha_factors]

    (_, (doobs)) = scan(step, (doob_initial, rng), params)
    # jacobians = jacobians[::-1, :, :]

    if mode == "average":
        # this should be (ts[-1] - ts[:-1]) instead of 1 maybe
        doobs = doobs[::-1, :]
        doobs = doobs / (ts[-1] - ts[:-1, None])
    elif mode == "optimal":
        # WARNING: This currently only works if ts are equidistant
        # Otherwise we need more something like a convolution
        # Since in fact the last one of the doobs is the one which got multiplied so often
        # not the first one as it is now

        def update_sum(carry, params):
            # carry represents "sum"in each iteration
            alpha, dt = params
            new_sum = dt + alpha * carry
            return new_sum, new_sum  # (updated carry, value to store in output)

        _, alpha_integrated = scan(update_sum, 0.0, [alpha_factors, dts])

        # prods = jnp.cumsum(prods)
        doobs /= alpha_integrated[:, None]
        doobs = doobs[::-1, :]

    elif mode == "last":
        # doobs are in wrong order
        doobs /= dts[0]
        doobs = doobs[::-1, :]
    elif mode == "first":
        doobs /= dts[:, None]
        doobs = doobs[::-1, :]
    else:
        raise Exception(f"Mode '{mode}' not supported")
    # elif mode == "soc":
    #     doobs = doobs[::-1, :]

    assert doobs.shape == (sample_path.shape[0] - 1, sample_path.shape[1])
    return doobs
