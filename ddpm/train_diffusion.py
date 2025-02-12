import functools
import pickle
from collections import namedtuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import tqdm

from sdes import run_sde_euler_maryuama, sdes

TrainingOptions = namedtuple("training", ["batch_size", "num_epochs", "batches_per_epoch"])


def plot_images(images, save_name):
    """
    Plot a list of images in a row and save the plot.

    Args:
        images: A list of images.
        save_name: The name of the file to save the plot.
    """
    n = len(images)
    fig, axs = plt.subplots(1, n)
    for i in range(n):
        axs[i].imshow(images[i], cmap="gray")
        axs[i].axis("off")
    plt.savefig(save_name)


def data_iterator(key, batch_size, data):
    """
    Given a dataset in form of arrays, returns an iterator that yields batches of data of size batch_size.

    Args:
        key: JAX random key.
        batch_size: The size of the batch.
        data: The dataset, a jax array with shape (dataset_size, height, width, channels).
    """

    dataset_size = data.shape[0]
    indices = jnp.arange(dataset_size)
    while True:
        key, subkey = jax.random.split(key)
        perm = jax.random.permutation(subkey, indices)
        start = 0
        end = batch_size
        while end <= dataset_size:
            batch_perm = perm[start:end]
            yield data[batch_perm]
            start = end
            end = start + batch_size


def loss_fn(params, model, key, data, alphas):
    """
    Compute the loss function for the diffusion model.

    Args:
        params: Flax parameters of the model.
        model: Flax model.
        key: JAX random key.
        data: The batch of data with shape (batch_size, height, width, channels).
        alphas: The alphas for the diffusion process where alphas[t] = prod_{i=1}^t (1 - betas[i]). shape (T,)
    """
    key, step_key = jax.random.split(key)

    alpha = jax.random.choice(step_key, alphas, (data.shape[0], 1, 1, 1))

    key, step_key = jax.random.split(key)
    noise = jax.random.normal(step_key, data.shape)
    noised_data = data * alpha**0.5 + noise * (1 - alpha) ** 0.5

    output = model.apply(params, noised_data, alpha.squeeze())
    loss = jnp.mean((noise - output) ** 2)

    return loss


@functools.partial(jax.jit, static_argnums=(4, 5))
def update_step(params, key, batch, opt_state, model, optimizer, alphas):
    val, grads = jax.value_and_grad(loss_fn)(params, model, key, batch, alphas)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return val, params, opt_state


def alphas_and_betas(beta_min, beta_max, N):
    """
    Compute the alphas and betas for the diffusion process.
    We set beta to be linearly spaced between beta_min and beta_max, and alphas[i]=prod_{j=1}^i (1 - betas[j]).

    Args:
        beta_min: The minimum value of beta. Should be >0.
        beta_max: The maximum value of beta. Should be <1.
    """
    betas = jnp.linspace(beta_min, beta_max, N)
    alphas = jnp.cumprod(1 - betas)
    return (
        alphas,
        betas,
    )


def sample(key, N_samples, alphas, betas, params, model, plot=True):
    """
    Sample from the diffusion model. Algorithm 2 from paper "Denoising Diffusion Probabilistic Models" by Ho et al.

    Args:
        key: JAX random key.
        N_samples: The number of samples to generate.
        alphas: The alphas of the model, defined as alphas[i]=prod_{j=1}^i (1 - betas[j]).
        betas: The betas of the model.
        params: The parameters of the model.
        model: The Flax model.
        plot: If True, plot the samples and the denoising process.
    """
    key, step_key = jax.random.split(key)
    noised_data = jax.random.normal(step_key, (N_samples, 28, 28, 1))
    plots = []

    pbar = tqdm.tqdm(
        range(len(betas)),
        desc="Sampling",
        leave=True,
        unit="steps",
        total=len(betas),
    )

    last_i = len(betas) - 1
    for i in pbar:
        beta = betas[-i]
        alpha = alphas[-i] * jnp.ones((noised_data.shape[0], 1, 1, 1))
        sigma = beta**0.5

        noise_guess = model.apply(params, noised_data, alpha.squeeze())
        key, step_key = jax.random.split(key)
        new_noise = jax.random.normal(step_key, noised_data.shape)
        noised_data = 1 / (1 - beta) ** 0.5 * (noised_data - beta / (1 - alpha) ** 0.5 * noise_guess)
        if not (i == last_i):
            noised_data += sigma * new_noise

        if plot and ((i % (len(betas) // 8) == 0) or (i == last_i)):
            plots.append(noised_data)

    return noised_data, jnp.asarray(plots)


def sample2(key, N_samples, alphas, betas, params, model, plot=True):
    sde = sdes.mnist_sde(alphas, betas, model, params)
    key, step_key = jax.random.split(key)
    noised_data = jax.random.normal(step_key, (N_samples, 28, 28, 1))

    ts = jnp.linspace(0, 1, len(betas))
    traj = run_sde_euler_maryuama.run_sde(key, sde, ts, noised_data)

    return traj, traj[-1]


def train_diffusion(key, data, model, training: TrainingOptions, alphas, betas):
    """
    Train the diffusion model. Algorithm 1 from paper "Denoising Diffusion Probabilistic Models" by Ho et al.

    Args:
        key: JAX random key.
        data: The dataset, a jax array with shape (dataset_size, height, width, channels).
        model: The Flax model.
        training: Training options instance containing batch_size, num_epochs and batches_per_epoch.
        alphas: The alphas for the diffusion process where alphas[t] = prod_{i=1}^t (1 - betas[i]). shape (T,)
    """
    batches = data_iterator(key, training.batch_size, data)

    dummy_x = jnp.zeros(shape=(training.batch_size, *data[0].shape))
    dummy_t = jnp.zeros(shape=(training.batch_size,))
    key, key_params, key_data = jax.random.split(jax.random.PRNGKey(0), 3)
    params = model.init(key_params, dummy_x, dummy_t)

    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)

    pbar = tqdm.tqdm(
        range(training.num_epochs),
        desc="Training",
        leave=True,
        unit="epoch",
        total=training.num_epochs,
    )

    for k in pbar:
        total_loss = 0
        for _ in range(training.batches_per_epoch):
            key, step_key = jax.random.split(key)
            batch = next(batches)
            loss, params, opt_state = update_step(params, step_key, batch, opt_state, model, optimizer, alphas)
            total_loss += loss
        epoch_loss = total_loss / training.batches_per_epoch
        pbar.set_postfix(Epoch=k + 1, loss=f"{epoch_loss:.4f}")
    pickle.dump(params, open("model_params", "wb"))
    return params
