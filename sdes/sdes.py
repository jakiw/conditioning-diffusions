import jax.numpy as jnp

from metrics.doob_h_1d import get_doob_drift


def bm_drift(t, x):
    return x * 0


def bm_sigma(t, x, dBt):
    return dBt


def bm_a(t, x, v):
    return v


def bm_sigma_transp_inv(t, x, dBt):
    return dBt


bm = (bm_drift, bm_sigma, bm_a, bm_sigma_transp_inv)


def double_well_sde(potential_height, y_obs):
    # only for debugging
    # def double_well_potential(x):
    #     return (x**2 - 1)**2 * potential_height

    def double_well_drift(t, x):
        return -4 * (x**3 - x) * potential_height

    double_well = (double_well_drift, bm_sigma, bm_a, bm_sigma_transp_inv)
    _control_xs = jnp.linspace(-4, 4, 1000)
    _control_ts = jnp.linspace(0, 1, 200)
    true_control, _ = get_doob_drift(double_well, y_obs, _control_ts, _control_xs)
    return double_well, true_control


def ou_sde(alpha, y_obs):
    def ou_drift(t, x):
        return alpha * x

    ou = (ou_drift, bm_sigma, bm_a, bm_sigma_transp_inv)
    _control_xs = jnp.linspace(-4, 4, 1000)
    _control_ts = jnp.linspace(0, 1, 200)
    true_control, _ = get_doob_drift(ou, y_obs, _control_ts, _control_xs)
    return ou, true_control


def sin_well_sde(potential_height, y_obs):
    # the potential should have minima at 1 and -1
    # so it is kind of cos()
    # then the drift is - the derivative of that, so its sin

    def drift(t, x):
        return potential_height * jnp.sin(x * 2 * jnp.pi)

    sde = (drift, bm_sigma, bm_a, bm_sigma_transp_inv)
    _control_xs = jnp.linspace(-4, 4, 1000)
    _control_ts = jnp.linspace(0, 1, 200)
    true_control, _ = get_doob_drift(sde, y_obs, _control_ts, _control_xs)
    return sde, true_control


def dm_toy_sde(t_initial, t_final, y_obs):
    from jax.scipy.special import logsumexp

    @partial(grad, argnums=1)
    def nabla_log_potential(t, x):
        p = -jnp.array([(x - 1) ** 2, (x + 1) ** 2]) / (2 * t)
        return logsumexp(p)

    def drift(t, x):
        t = 1 - t
        t = t_initial + (t_final * t)
        return nabla_log_potential(t, x)

    sde = (drift, bm_sigma, bm_a, bm_sigma_transp_inv)
    _control_xs = jnp.linspace(-4, 4, 1000)
    _control_ts = jnp.linspace(0, 1, 200)
    true_control, _ = get_doob_drift(sde, y_obs, _control_ts, _control_xs)
    return sde, true_control


# ts = jnp.linspace(0, 1, 200)
# xs = jnp.linspace(-4, 4, 2000)
# double_well_conditioned, _ld = get_doob_drift(double_well, y_obs, ts, xs)
