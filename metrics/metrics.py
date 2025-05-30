from functools import partial

import jax.numpy as jnp
from jax import jit, random, vmap

from helpers import apply_nn_drift_sde, apply_nn_drift_sde_y_free, apply_nn_drift_y_free
from sdes import sdes
from sdes.run_sde_euler_maryuama import run_sde


def compare_with_true_drift(t, x, y, control, true_drift):
    control_eval = control(t, x, y)
    true_eval = true_drift(t, x)
    err = jnp.sum((true_eval - control_eval) ** 2)
    return err


def compare_with_true_drift_along_path(ts, path, y, control, true_drift):
    errors = vmap(compare_with_true_drift, in_axes=(0, 0, None, None, None))(
        ts[:-1], path[:-1, :], y, control, true_drift
    )
    dts = ts[1:] - ts[:-1]
    integral = jnp.sum(errors * dts)
    return integral


@partial(jit, static_argnames=["sde", "nn_model", "true_control"])
def kl_metrics(rng, sde, ts, nn_model, nn_params, initial_samples, true_control, y_obs):
    # Computes KL(Generated | Truth)
    # drift, sigma, a, sigma_transp_inv = sde
    rngs = random.split(rng, initial_samples.shape[0])

    nn_drift = apply_nn_drift_y_free(nn_model, nn_params)
    # nn_sde = apply_nn_drift_sde(sde, nn_model, nn_params)

    conditioned_sde = sdes.conditioned_sde(sde, true_control)

    paths, _, __ = vmap(run_sde, in_axes=(0, None, None, 0), out_axes=(0))(rngs, conditioned_sde, ts, initial_samples)
    # ys = jnp.tile(y_obs, (paths.shape[1] - 1, 1))
    err = vmap(compare_with_true_drift_along_path, in_axes=(None, 0, None, None, None))(
        ts, paths, y_obs, nn_drift, true_control
    )
    return jnp.mean(err) ** 0.5


def endpoint_distance_amortized(rng, sde, ts, true_control, initial_samples):
    # Making sure different brownian motions are used here and
    # in the generatino of the paths using the control
    # otherwise the uncontrolled paths with these brownian motions already reach the target
    rng, srngs = random.split(rng)
    rng, srngs = random.split(rng)
    rngs = random.split(srngs, initial_samples.shape[0])
    paths, _, __ = vmap(run_sde, in_axes=(0, None, None, 0), out_axes=(0))(rngs, sde, ts, initial_samples)

    observations = paths[:, -1, :]

    @partial(jit, static_argnames=["sde", "nn_model", "true_control"])
    def metric(rng, sde, ts, nn_model, nn_params, initial_samples, true_control, y_obs):
        conditioned_sde = apply_nn_drift_sde_y_free(sde, nn_model, nn_params)

        rngs = random.split(rng, initial_samples.shape[0])
        paths, _, __ = vmap(run_sde, in_axes=(0, None, None, 0, 0), out_axes=(0))(
            rngs, conditioned_sde, ts, initial_samples, observations
        )

        endpoint_distances = jnp.sum((paths[:, -1, :] - observations) ** 2, axis=1)
        endpoint_distances = jnp.mean(endpoint_distances**0.5)
        return endpoint_distances

    return metric


def endpoint_distance(rng, sde, ts, true_control, initial_samples):
    # Making sure different brownian motions are used here and
    # in the generatino of the paths using the control
    # otherwise the uncontrolled paths with these brownian motions already reach the target
    rng, srngs = random.split(rng)
    rng, srngs = random.split(rng)
    rngs = random.split(srngs, initial_samples.shape[0])
    paths, _, __ = vmap(run_sde, in_axes=(0, None, None, 0), out_axes=(0))(rngs, sde, ts, initial_samples)

    @partial(jit, static_argnames=["sde", "nn_model", "true_control"])
    def metric(rng, sde, ts, nn_model, nn_params, initial_samples, true_control, y_obs):
        conditioned_sde = apply_nn_drift_sde(sde, nn_model, nn_params, y_obs)

        rngs = random.split(rng, initial_samples.shape[0])
        paths, _, __ = vmap(run_sde, in_axes=(0, None, None, 0), out_axes=(0))(
            rngs, conditioned_sde, ts, initial_samples
        )

        endpoint_distances = jnp.sum((paths[:, -1, :] - y_obs[None, :]) ** 2, axis=1)
        endpoint_distances = jnp.mean(endpoint_distances**0.5)
        return endpoint_distances

    return metric


def amortized_mean_var_metric(rng, sde, ts, true_control, initial_samples):
    # Making sure different brownian motions are used here and
    # in the generatino of the paths using the control
    # otherwise the uncontrolled paths with these brownian motions already reach the target
    rng, srngs = random.split(rng)
    rng, srngs = random.split(rng)
    rngs = random.split(srngs, initial_samples.shape[0])
    paths, _, __ = vmap(run_sde, in_axes=(0, None, None, 0), out_axes=(0))(rngs, sde, ts, initial_samples)

    means = jnp.mean(paths, axis=0)
    # transposed_paths = jnp.transpose(paths_conditioned_truth, axes=(2, 1, 0))
    # vars = vmap(jnp.paths, in_axes=(1))(paths_conditioned_truth.T)
    vars = jnp.var(paths, axis=0)
    observations = paths[:, -1, :]

    @partial(jit, static_argnames=["sde", "nn_model", "true_control"])
    def metric(rng, sde, ts, nn_model, nn_params, initial_samples, true_control, y_obs):
        # nn_control = lambda t, x, y: nn_model.apply(nn_params, t, x, y)
        # nn_control_and_sde_drift = lambda t, x: nn_control(t, x, y_obs) + drift(t, x)
        # conditioned_sde = (nn_control_and_sde_drift, sigma, a, sigma_transp_inv)
        conditioned_sde = apply_nn_drift_sde_y_free(sde, nn_model, nn_params)

        rngs = random.split(rng, initial_samples.shape[0])
        paths, _, __ = vmap(run_sde, in_axes=(0, None, None, 0, 0), out_axes=(0))(
            rngs, conditioned_sde, ts, initial_samples, observations
        )
        means_gen = jnp.mean(paths, axis=0)
        # vars_gen = jnp.var(paths, axis=0)
        vars_gen = jnp.var(paths, axis=0)

        mean_error = jnp.mean((means - means_gen) ** 2, axis=1)

        var_error = jnp.mean((vars**0.5 - vars_gen**0.5) ** 2, axis=1)

        error = jnp.mean(jnp.sqrt(mean_error + var_error))

        return error

    return metric


def mean_var_metric(rng, sde, ts, true_control, initial_samples):
    sde_conditioned_truth = sdes.conditioned_sde(sde, true_control)

    rngs = random.split(rng, initial_samples.shape[0])
    paths_conditioned_truth, _, __ = vmap(run_sde, in_axes=(0, None, None, 0), out_axes=(0))(
        rngs, sde_conditioned_truth, ts, initial_samples
    )

    means = jnp.mean(paths_conditioned_truth, axis=0)
    vars = jnp.var(paths_conditioned_truth, axis=0)

    @partial(jit, static_argnames=["sde", "nn_model", "true_control"])
    def metric(rng, sde, ts, nn_model, nn_params, initial_samples, true_control, y_obs):
        conditioned_sde = apply_nn_drift_sde(sde, nn_model, nn_params, y_obs)

        rngs = random.split(rng, initial_samples.shape[0])
        paths, _, __ = vmap(run_sde, in_axes=(0, None, None, 0), out_axes=(0))(
            rngs, conditioned_sde, ts, initial_samples
        )

        means_gen = jnp.mean(paths, axis=0)
        vars_gen = jnp.var(paths, axis=0)

        mean_error = jnp.mean((means - means_gen) ** 2, axis=1)
        mean_error = mean_error**0.5

        var_error = jnp.mean((vars**0.5 - vars_gen**0.5) ** 2, axis=1)
        var_error = var_error**0.5

        error = jnp.mean(jnp.sqrt(mean_error + var_error))

        return error

    return metric



def get_energy_metric(sde, energy, ts):

    @partial(jit, static_argnames=["nn_model"])
    def metric(rng, nn_model, nn_params, initial, target):
        conditioned_sde = apply_nn_drift_sde(sde, nn_model, nn_params, target)
        rngs = random.split(rng, initial.shape[0])

        paths, _, __ = vmap(run_sde, in_axes=(0, None, None, 0), out_axes=(0))(
            rngs, conditioned_sde, ts, initial
        )
        # vmap over batch and time dimension
        energy_gen = vmap(vmap(energy))(paths[:, :, :66])
        mean_energy = jnp.mean(energy_gen)

        return mean_energy

    return metric