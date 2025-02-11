import jax
import jax.numpy as jnp
from jax import vmap

# Take a neural network model and reparametrizes it as a control function for an SDE


def apply_nn_drift(nn_model, nn_params, y_obs, detach=False):
    if detach == True:
        used_params = jax.lax.stop_gradient(nn_params)
    else:
        used_params = nn_params

    def nn_control(t, x):
      return nn_model.apply(used_params, t, x, y_obs) * 1/jnp.sqrt(1 - t)

    return nn_control


def apply_nn_drift_y_free(nn_model, nn_params, detach=False):
    if detach == True:
        used_params = jax.lax.stop_gradient(nn_params)
    else:
        used_params = nn_params

    def nn_control(t, x, y):
      return nn_model.apply(used_params, t, x, y) * 1/jnp.sqrt(1 - t)

    return nn_control

def apply_nn_drift_sde(sde, nn_model, nn_params, y_obs, detach=False):
    drift, sigma, a, sigma_transp_inv = sde

    nn_control = apply_nn_drift(nn_model, nn_params, y_obs, detach=detach)
    def nn_control_with_sde_drift(t, x):
      return nn_control(t, x) + drift(t, x)

    nn_sde = (nn_control_with_sde_drift, sigma, a, sigma_transp_inv)
    return nn_sde

def apply_nn_drift_sde_y_free(sde, nn_model, nn_params, detach=False):
    drift, sigma, a, sigma_transp_inv = sde

    nn_control = apply_nn_drift_y_free(nn_model, nn_params, detach=detach)
    def nn_control_with_sde_drift(t, x, y):
      return nn_control(t, x, y) + drift(t, x)

    nn_sde = (nn_control_with_sde_drift, sigma, a, sigma_transp_inv)
    return nn_sde




# VMAPPING THE SDE AND CONTROLS

def vmap_sde_dimension(sde):
    drift, sigma, a, sigma_transp_inv = sde
    drift_D = vmap(drift, in_axes=(None, 0))
    sigma_D = vmap(sigma, in_axes=(None, 0, 0))
    a_D = vmap(sigma, in_axes=(None, 0, 0))
    sigma_transp_inv_D = vmap(sigma_transp_inv, in_axes=(None, 0, 0))
    sde_D = (drift_D, sigma_D, a_D, sigma_transp_inv_D)
    return sde_D

def vmap_control_only_first_dimension(control, D):

    def new_control(t, x):
        zeros = jnp.zeros(D)
        c = control(t, x[0])
        zeros = zeros.at[0].set(c)
        return zeros
    
    return new_control



# So we can save the args and kwargs we pass to a function in a dict and serialize it
# During execution then args and kwargs are applied:

def apply_function(data):
    function = data["function"]
    
    #get args from data and if it doesnt exist set to empty tuple
    args = data["args"] if "args" in data else ()
    kwargs = data["kwargs"] if "kwargs" in data else {}
    return function(*args, **kwargs)