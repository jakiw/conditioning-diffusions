import jax
import jax.numpy as jnp

from helpers import vmap_control_only_first_dimension, vmap_sde_dimension
from sdes.sdes import dm_toy_sde, double_well_sde, ou_sde, sin_well_sde


def dm_toy_problem(D=1):
    y_obs = -1
    y_obs_arr = jnp.ones(D) * y_obs
    sde, control = dm_toy_sde(0.005, 2, y_obs)

    sde = vmap_sde_dimension(sde)
    # control = vmap(control, in_axes=(None, 0))
    control = vmap_control_only_first_dimension(control, D)

    y_initial_validation = jnp.ones(D) * y_obs
    y_initial_validation = y_initial_validation.at[0].set(1)

    return sde, control, y_obs_arr, y_initial_validation, f"Diffusion Model Problem"


def sin_toy_problem(potential_height, D=1):
    y_obs = -1
    y_obs_arr = jnp.ones(D) * y_obs
    sde, control = sin_well_sde(potential_height, y_obs)

    sde = vmap_sde_dimension(sde)
    control = vmap_control_only_first_dimension(control, D)

    y_initial_validation = jnp.ones(D) * y_obs
    y_initial_validation = y_initial_validation.at[0].set(1)

    return sde, control, y_obs_arr, y_initial_validation, f"Sin Problem ({potential_height})"


def double_well_toy_problem(potential_height, D=1):
    y_obs = -1
    y_obs_arr = jnp.ones(D) * y_obs
    sde, control = double_well_sde(potential_height, y_obs)

    sde = vmap_sde_dimension(sde)
    control = vmap_control_only_first_dimension(control, D)

    y_initial_validation = jnp.ones(D) * y_obs
    y_initial_validation = y_initial_validation.at[0].set(1)

    return sde, control, y_obs_arr, y_initial_validation, f"Double Well Problem ({potential_height})"


def double_well_toy_problem_opening(potential_height, D=1):
    y_obs = -1
    y_obs_arr = jnp.ones(D) * y_obs
    sde, control = double_well_sde(3, y_obs)

    sde = vmap_sde_dimension(sde)

    drift, sigma, a, sigma_transp_inv = sde

    def new_drift(t, x):
        left = 0.4
        right = 0.6
        sharpness = 10
        indicator_left_right = jax.nn.sigmoid((x - left) * sharpness) - jax.nn.sigmoid((x - right) * sharpness)
        return drift(t, x) * (1 - indicator_left_right)

    sde = (new_drift, sigma, a, sigma_transp_inv)

    control = vmap_control_only_first_dimension(control, D)

    y_initial_validation = jnp.ones(D) * y_obs
    y_initial_validation = y_initial_validation.at[0].set(1)

    return sde, control, y_obs_arr, y_initial_validation, f"Double Well Problem With Pathway ({potential_height})"


def ou_toy_problem(alpha, D=1):
    y_obs = -1
    y_obs_arr = jnp.ones(D) * y_obs
    sde, control = ou_sde(alpha, y_obs)

    sde = vmap_sde_dimension(sde)
    control = vmap_control_only_first_dimension(control, D)

    y_initial_validation = jnp.ones(D) * y_obs
    y_initial_validation = y_initial_validation.at[0].set(1)

    return sde, control, y_obs_arr, y_initial_validation, f"OU Problem ({alpha})"


# def slow_circle_toy_problem():
#     #only for debugging
#     # def double_well_potential(x):
#     #     return (x**2 - 1)**2 * potential_height

#     def drift(t, x):
#         return 0

#     def sigma_value(t, x):
#         norm_x = jnp.sum(x**2)**0.5
#         left = -0.5
#         right = 0.5
#         sharpness = 10
#         indicator_left_right = jax.nn.sigmoid((norm_x - left)*sharpness) - jax.nn.sigmoid((norm_x - right)*sharpness)
#         slow_circle = 0.05 + 1 - indicator_left_right
#         return slow_circle

#     def sigma(t, x, dBt):
#         s = sigma_value(t, x)
#         return s * dBt

#     def a(t, x, dBt):
#         s = sigma_value(t, x)
#         return s**2 * dBt

#     def sigma_transp_inv(t, x, dBt):
#         s = sigma_value(t, x)
#         return dBt / s

#     sde = (drift, sigma, a, sigma_transp_inv)

#     y_obs = -1
#     y_obs_arr = jnp.array([1.0, 0.0])
#     y_initial_validation = jnp.array([-1.0, 0.0])

#     y_initial_validation = jnp.ones(D) * y_obs
#     y_initial_validation = y_initial_validation.at[0].set(1)

#     return sde, drift, y_obs_arr, y_initial_validation, f"Slow Circle"
