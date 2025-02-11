import dataclasses
import functools
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp

from metrics.doob_h_1d import get_doob_drift


@dataclasses.dataclass(frozen=True, eq=True)
class SDE:
    drift: Callable
    sigma: Callable
    covariance: Callable
    sigma_transp_inv: Callable


def bm():
    def bm_drift(t, x):
        return x * 0

    return SDE(bm_drift, bm_sigma, bm_a, bm_sigma_transp_inv)


def double_well_sde(potential_height):
    def double_well_drift(t, x):
        return -4 * (x**3 - x) * potential_height

    return SDE(double_well_drift, bm_sigma, bm_a, bm_sigma_transp_inv)


def ou_sde(alpha):
    def ou_drift(t, x):
        return alpha * x

    return SDE(ou_drift, bm_sigma, bm_a, bm_sigma_transp_inv)


def sin_well_sde(potential_height):
    # the potential should have minima at 1 and -1
    # so it is kind of cos()
    # then the drift is - the derivative of that, so its sin

    def drift(t, x):
        return potential_height * jnp.sin(x * 2 * jnp.pi)

    return SDE(drift, bm_sigma, bm_a, bm_sigma_transp_inv)


def dm_toy_sde(t_initial, t_final):
    @functools.partial(jax.grad, argnums=1)
    def nabla_log_potential(t, x):
        p = -jnp.array([(x - 1) ** 2, (x + 1) ** 2]) / (2 * t)
        return jax.scipy.special.logsumexp(p)

    def drift(t, x):
        t = 1 - t
        t = t_initial + (t_final * t)
        return nabla_log_potential(t, x)

    return SDE(drift, bm_sigma, bm_a, bm_sigma_transp_inv)


def conditioned_sde(sde, true_control):
    def true_conditioned_drift(t_, x_):
        return sde.drift(t_, x_) + sde.covariance(t_, x_, true_control(t_, x_))

    return SDE(true_conditioned_drift, sde.sigma, sde.covariance, sde.sigma_transp_inv)


def true_control(sde, y_obs):
    _control_xs = jnp.linspace(-4, 4, 1000)
    _control_ts = jnp.linspace(0, 1, 200)
    true_control, _ = get_doob_drift(sde, y_obs, _control_ts, _control_xs)
    return true_control


def bm_sigma(t, x, dBt):
    return dBt


def bm_a(t, x, v):
    return v


def bm_sigma_transp_inv(t, x, dBt):
    return dBt
