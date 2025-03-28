from flow_matching_jax.flow_matching_sampling import get_sde as get_sde_fm
from flow_matching_jax.flow_matching_sampling import SDE as SDE_FM
from flow_matching_jax.models.get_model import TrainState
from flow_matching_jax.train import restore_checkpoint, create_train_state
from sdes.sdes import SDE
import jax
from jax import random
import jax.numpy as jnp
import orbax
import numpy as onp
from matplotlib import pyplot as plt
import wandb
from helpers import apply_nn_drift_sde, apply_nn_drift_sde_y_free, apply_nn_drift_y_free
from sdes import sdes
from sdes.run_sde_euler_maryuama import run_sde
from jax import vmap
import numpy as onp


def restore_checkpoint_from_config_directory(config, workdir):
    rng = jax.random.PRNGKey(config.seed)
    state = create_train_state(rng, config)
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=2, create=True)
    checkpoint_manager = orbax.checkpoint.CheckpointManager(
        workdir, orbax_checkpointer, options)
    state = restore_checkpoint(state, checkpoint_manager)
    return config, state


def log_image_grid(images, step, label="image_grid", cmap=None):
    N, W, H, C = images.shape
    n = int(jnp.sqrt(N))
    assert n * n == N, "N must be a perfect square"

    fig, axs = plt.subplots(n, n, figsize=(10, 10))
    axs = axs.reshape(-1)  # flatten in case axs is 2D

    #set figure title
    fig.suptitle(label)
    #set figure to tight
    fig.tight_layout()

    for i in range(N):
        axs[i].imshow(images[i], cmap=cmap)
        axs[i].axis('off')

    for ax in axs[N:]:
        ax.axis('off')

    # save figure to workdir/label with dpi 300 and tight bbox
    fig.savefig(f"workdir/{label}_{step}.png", dpi=300, bbox_inches='tight')

    wandb.log({label: wandb.Image(fig), "step": step})
    plt.close(fig)


def get_problem(config, workdir, ts, get_obs = None, N_samples_eval=16):
    if get_obs is None:
        def get_obs(x):
            return x
    _, state = restore_checkpoint_from_config_directory(config, workdir)
    h = 1/45
    def my_sigma(t):
        return jnp.sqrt(2 * (1 - t + h) / (t + h))
        # return 0.5
    
    # sigmas.append(lambda t: )

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
    

    rng = random.PRNGKey(1995)
    y_init_eval = random.normal(rng, config.data.shape)
    y_init_eval_arr = jnp.repeat(y_init_eval[jnp.newaxis, ...], N_samples_eval, axis=0)


    def covariance(t, x, dBt):
        return dBt * my_sigma(t)**2
    
    def sigma_transpose_inv(t, x, dBt):
        return dBt / my_sigma(t)

    sde = SDE(drift, sigma, covariance, sigma_transpose_inv)
    
    X = jnp.ones(config.data.shape) * 0.5
    X = jnp.array(onp.load("/home/ubuntu/WashingtonMain/conditioning-diffusions/problems/sample.npy"))
    y_obs = get_obs(X)


    def sample_metric(rng, nn_model, nn_params, y_init_eval, y_obs, step):
        conditioned_sde = apply_nn_drift_sde(sde, nn_model, nn_params, y_obs)

        rngs = random.split(rng, y_init_eval_arr.shape[0])
        paths, _, __ = vmap(run_sde, in_axes=(0, None, None, 0), out_axes=(0))(
            rngs, conditioned_sde, ts, y_init_eval_arr
        )
        samples = paths[:, -1, :]
        samples = samples.at[0, :].set(X)
        name = f"samples"
        log_image_grid(samples, step, label=name, cmap="Greys")

        observations = vmap(get_obs)(samples)
        name = f"observations"
        log_image_grid(observations, step, label=name, cmap="Greys")

        return 0

    metrics = {
        "sample_metric": sample_metric
    }

    return sde, metrics, get_obs, y_obs, y_init_eval, f"Diffusion Model"

