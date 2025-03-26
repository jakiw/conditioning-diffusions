from flow_matching_jax.flow_matching_sampling import get_sde
from flow_matching_jax.models.get_model import TrainState
from flow_matching_jax.train import restore_checkpoint, create_train_state
import jax
from jax import random
import jax.numpy as jnp
import orbax


def restore_checkpoint(state, checkpoint_manager):
    step = checkpoint_manager.latest_step()  
    if step is None:
        return state
    
    restore_args = flax.training.orbax_utils.restore_args_from_target(state, mesh=None)
    state = checkpoint_manager.restore(step, items=state, restore_kwargs={'restore_args': restore_args})
    return state


def restore_checkpoint_from_config_directory(config, workdir):
    rng = jax.random.PRNGKey(config.seed)
    state = create_train_state(rng, config)
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=2, create=True)
    checkpoint_manager = orbax.checkpoint.CheckpointManager(
        workdir, orbax_checkpointer, options)
    state = restore_checkpoint(state, checkpoint_manager)
    return config, state


def get_sde(checkpoint_file, D=1):
    y_obs = -1
    y_obs_arr = jnp.ones(D) * y_obs
    sde, control = ou_sde(alpha)
    control = true_control(sde, y_obs)

    sde = vmap_sde_dimension(sde)
    control = vmap_control_only_first_dimension(control, D)

    y_initial_validation = jnp.ones(D) * y_obs
    y_initial_validation = y_initial_validation.at[0].set(1)

    return sde, control, y_obs_arr, y_initial_validation, f"OU Problem ({alpha})"
