from flow_matching_jax.flow_matching_sampling import get_sde as get_sde_fm
from flow_matching_jax.flow_matching_sampling import SDE as SDE_FM
from flow_matching_jax.models.get_model import TrainState
from flow_matching_jax.train import restore_checkpoint, create_train_state
from sdes.sdes import SDE
import jax
from jax import random
import jax.numpy as jnp
import orbax


def restore_checkpoint_from_config_directory(config, workdir):
    rng = jax.random.PRNGKey(config.seed)
    state = create_train_state(rng, config)
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=2, create=True)
    checkpoint_manager = orbax.checkpoint.CheckpointManager(
        workdir, orbax_checkpointer, options)
    state = restore_checkpoint(state, checkpoint_manager)
    return config, state


def get_problem(config, workdir):
    _, state = restore_checkpoint_from_config_directory(config, workdir)
    def my_sigma(t):
        return 0.5
    
    sde_fm = get_sde_fm(state, my_sigma, True)

    def unbatched_drift(t, x):
        # x = jnp.expand_dims(x, 0)
        # t = jnp.ones(1) * t
        out = sde_fm.drift(t, x)
        # out = jnp.squeeze(out, 0)
        return out
    
    drift = unbatched_drift
    def sigma(t, x, dBt):
        return dBt * my_sigma(t)
    

    def covariance(t, x, dBt):
        return dBt * my_sigma(t)**2
    
    def sigma_transpose_inv(t, x, dBt):
        return dBt / my_sigma(t)

    sde = SDE(drift, sigma, covariance, sigma_transpose_inv)
    
    y_obs = jnp.ones(config.data.shape) * 0.5

    control = None

    rng = random.PRNGKey(1995)
    y_initial_validation = random.normal(rng, config.data.shape)

    return sde, control, y_obs, y_initial_validation, f"Diffusion Model"

