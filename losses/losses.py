from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
from jax import grad, random, vmap
from jax.lax import scan

from helpers import apply_nn_drift_sde
from losses.solve_adjoint_sde import solve_J_equation_2
from sdes.run_sde_euler_maryuama import run_sde

## Gaussian Simple


@partial(grad, argnums=0)
def nabla_log_potential(x, x_, t, dt, sde):
    drift, sigma, a, sigma_transp_inv = sde
    return -jnp.sum((x + dt * drift(t, x) - x_) ** 2) / (2 * dt * sigma(t, x))


def get_loss_single_path(rng, sde, nn_model, nn_params, ts, initial_sample):
    rng, srng = random.split(rng)
    sample_path, _, __ = run_sde(srng, sde, ts, initial_sample, noise_last_step=True)
    dts = ts[1:] - ts[:-1]
    x = sample_path[:-1, :]
    x_ = sample_path[1:, :]
    y = sample_path[-1, :]
    # ys = jnp.tile(y, (x.shape[0], 1))
    predictions = vmap(nn_model.apply, in_axes=(None, 0, 0, None))(nn_params, ts[:-1], sample_path[:-1], y)
    lp = vmap(nabla_log_potential, in_axes=(0, 0, 0, 0, None))(x, x_, ts[:-1], dts, sde)
    # jax.lax.cond(jnp.any(jnp.isnan(lp)), lambda: jprint("Infinity"), lambda: None)

    t_factor = (1 - ts[:-1, None]) ** 0.5
    error = predictions - lp * t_factor
    error = jnp.mean(error**2)
    # return error, sample, predictions, lp
    return error, sample_path, lp


def get_loss(rng, sde, nn_model, nn_params, ts, initial_samples, _y_obs):
    rngs = random.split(rng, initial_samples.shape[0])
    f = lambda rng_, s_: get_loss_single_path(rng_, sde, nn_model, nn_params, ts, s_)
    errors = vmap(f, in_axes=(0, 0), out_axes=(0))(rngs, initial_samples)
    return jnp.mean(errors)


def get_loss_nik_single_path(rng, sde, nn_model, nn_params, ts, initial_sample, **kwargs):
    rng, srng = random.split(rng)
    drift, sigma, a, sigma_transp_inv = sde
    sample_path, _drift_evals, dBts = run_sde(srng, sde, ts, initial_sample, noise_last_step=True)

    rng, srng = random.split(rng)
    doobs = solve_J_equation_2(srng, sde, ts, sample_path, dBts, **kwargs)
    y = sample_path[-1, :]
    predictions = vmap(nn_model.apply, in_axes=(None, 0, 0, None))(nn_params, ts[:-1], sample_path[:-1], y)
    t_factor = (1 - ts[:-1, None]) ** 0.5

    a_times_doobs = vmap(a, in_axes=(0, 0, 0))(ts[:-1], sample_path[:-1], doobs)
    error = predictions - a_times_doobs * t_factor
    # error = (predictions/t_factor - doobs) * t_factor
    # error = predictions/t_factor - doobs
    error = jnp.mean(error**2)
    # return error, sample, predictions, lp
    return error  # , sample_path, doobs, _jacobians


def get_loss_nik(rng, sde, nn_model, nn_params, ts, initial_samples, _y_obs, **kwargs):
    rngs = random.split(rng, initial_samples.shape[0])
    f = lambda rng_, initial_sample_: get_loss_nik_single_path(
        rng_, sde, nn_model, nn_params, ts, initial_sample_, **kwargs
    )
    errors = vmap(f, in_axes=(0, 0), out_axes=(0))(rngs, initial_samples)
    # breakpoint_on_nan(errors)
    return jnp.mean(errors)


def get_loss_nik_single_path_consistency(rng, sde, nn_model, nn_params, ts, initial_sample, y_obs, **kwargs):
    rng, srng = random.split(rng)
    # print("Loss single path")
    drift, sigma, a, sigma_transp_inv = sde

    # todo: detach gradients
    # def nn_control(t, x):
    #   detached_params = jax.lax.stop_gradient(nn_params)
    #   return nn_model.apply(detached_params, t, x, y_obs) * 1/jnp.sqrt(1 - t)

    # def nn_control_with_sde_drift(t, x):
    #   return nn_control(t, x) + drift(t, x)

    nn_sde = apply_nn_drift_sde(sde, nn_model, nn_params, detach=True)

    # nn_sde = (nn_control_with_sde_drift, sigma, a, sigma_transp_inv)

    # TODO: Am I taking derivative with respect to the path here too? I probably should not I think
    # WARNING: The control evals from this function are scaled by 1/sqrt{1 - t} so they need to go into the loss differently
    # Also they are detached now, so I just create new one
    sample_path, _, sample_path_dBts = run_sde(srng, nn_sde, ts, initial_sample, noise_last_step=False)
    # jprint("It has nans: {x}", x=jnp.any(jnp.isnan(sample_path)))

    # sample_path_detached = sample_path
    # Not sure if this is necessary but maybe in case the dBts are procued in some
    # complicated matter later which includes the drift to get the conditioned brownian motion

    rng, srng = random.split(rng)
    # control_evals = vmap(nn_control, in_axes=(0, 0))(ts[:-1], sample_path[:-1, :])
    dts = ts[1:] - ts[:-1]
    t_factor = (1 - ts[:-1, None]) ** 0.5
    # dBts += control_evals * dts[:, None]

    control_evals_non_scaled = vmap(nn_model.apply, in_axes=(None, 0, 0, None))(
        nn_params, ts[:-1], sample_path[:-1, :], y_obs
    )
    control_evals_scaled = control_evals_non_scaled * 1 / t_factor

    dBts_conditioned = control_evals_scaled * dts[:, None]  # + sample_path_dBts

    last_step_prediction = sample_path[-2, :] + dts[-1] * control_evals_scaled[-1]
    last_step_error = y_obs - last_step_prediction
    dBts_conditioned = dBts_conditioned.at[-1, :].set(last_step_error)

    doobs, jacobians = solve_J_equation_2(srng, sde, ts, sample_path, dBts_conditioned, **kwargs)
    # y = sample_path[-1, :]
    # ys = jnp.tile(y, (sample_path.shape[0]-1, 1))

    error = control_evals_non_scaled - doobs * t_factor
    # error = (predictions/t_factor - doobs) * t_factor
    # error = predictions/t_factor - doobs
    if "lct" in kwargs:
        LCT = kwargs["lct"]
    else:
        LCT = 0

    if LCT > 0:
        error = jnp.minimum(LCT, error)

    error = jnp.mean(error**2)
    # return error, sample, predictions, lp
    return error  # , sample_path, doobs, jacobians


def get_loss_nik_consistency(rng, sde, nn_model, nn_params, ts, initial_samples, y_obs, **kwargs):
    rngs = random.split(rng, initial_samples.shape[0])
    f = lambda rng_, initial_sample_: get_loss_nik_single_path_consistency(
        rng_, sde, nn_model, nn_params, ts, initial_sample_, y_obs, **kwargs
    )
    errors = vmap(f, in_axes=(0, 0), out_axes=(0))(rngs, initial_samples)
    # breakpoint_on_nan(errors)
    return jnp.mean(errors)


def get_loss_reparametrization_trick_single_trajectory(rng, sde, nn_model, nn_params, ts, initial_samples, **kwargs):
    drift, sigma, a, sigma_transp_inv = sde

    def transition_approx(M: Callable, scale: Callable, drift: Callable):
        dz_drift = jax.jacfwd(drift)
        ds_M = jax.jacfwd(M)

        def pathwise_score_fn(times: jax.Array, sample_path: jax.Array, dBts: jax.Array):
            def k_term(t_k, x_k, dB_k):
                return -ds_M(t_k) * dB_k + M(t_k) * (dz_drift(t_k, x_k).T @ dB_k)

            k_transition = jax.vmap(k_term, in_axes=(0, 0, 0))(times, sample_path, dBts)
            transition = jnp.cumsum(k_transition[::-1], axis=0)[::-1]
            scale_factor = jax.vmap(scale)(times)
            return scale_factor[:, None] * transition

        return pathwise_score_fn

    M = lambda t: (ts[-1] - t)
    scale = lambda s: 1 / (ts[-1] - s)
    reparam_doobs_fn = transition_approx(M, scale, drift)

    rng, srng = random.split(rng)
    sample_path, _, dBts = run_sde(srng, sde, ts, initial_samples, noise_last_step=True)

    y = sample_path[-1, :]
    # t = ts[:-1]
    # traj = trajectory[:, :-1]

    # true_score = jax.vmap(reparam_doobs_fn, in_axes=(None, None, None))(ts[:-1], sample_path[:-1, :], dBts)
    true_score = reparam_doobs_fn(ts[:-1], sample_path[:-1, :], dBts)

    t_factor = (1 - ts[:-1]) ** 0.5

    prediction = vmap(nn_model.apply, in_axes=(None, 0, 0, None))(nn_params, ts[:-1], sample_path[:-1, :], y)

    # difference = true_score * t_factor - prediction
    difference = true_score * t_factor[:, None] - prediction
    # unweighted_norm = difference[:, None, :] @ difference[:, :, None]
    loss = jnp.mean(difference**2)
    return loss

    # grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    # (_loss, updates), grads = grad_fn(params)
    # opt_updates, opt_state = optimiser.update(grads, opt_state, params)
    # params = optax.apply_updates(params, opt_updates)
    # batch_stats = updates["batch_stats"]
    # return params, batch_stats, opt_state, _loss


def get_loss_reparametrization_trick(rng, sde, nn_model, nn_params, ts, initial_samples, y_obs):
    rngs = random.split(rng, initial_samples.shape[0])
    f = lambda rng_, initial_sample_: get_loss_reparametrization_trick_single_trajectory(
        rng_, sde, nn_model, nn_params, ts, initial_sample_
    )
    errors = vmap(f, in_axes=(0, 0), out_axes=0)(rngs, initial_samples)
    return jnp.mean(errors)
