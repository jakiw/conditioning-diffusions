import pickle

import jax
import jax.numpy as jnp
import tensorflow_datasets as tfds
import train_diffusion
from unet import UNet

if __name__ == "__main__":
    key = jax.random.PRNGKey(0)

    # If true, train the model. If false, load the pickled model params from "model_params".
    to_train = False
    # If true, sample from the model. If false, load the pickled samples from "sampled" and "plots".
    to_sample = True

    T = 1000
    alphas, betas = train_diffusion.alphas_and_betas(1e-4, 0.02, T)
    model = UNet()

    if to_train:
        # Training options
        training = train_diffusion.TrainingOptions(num_epochs=100, batch_size=512, batches_per_epoch=118)

        # load mnist data
        mnist_data, info = tfds.load(name="mnist", batch_size=-1, with_info=True)
        mnist_data = tfds.as_numpy(mnist_data)
        train_data, test_data = mnist_data["train"], mnist_data["test"]
        train_images, train_labels = train_data["image"], train_data["label"]
        data = train_images / 255.0

        # Train the model
        params = train_diffusion.train_diffusion(key, data, model, training, alphas, betas)

    else:
        params = pickle.load(open("model_params", "rb"))

    if to_sample:
        # Sampled are the final samples from the model.
        # Plots are the intermediate samples during the denoising process, created if plot=True.

        sampled, plots = train_diffusion.sample2(key, 10, alphas, betas, params, model, plot=True)
        jnp.save("plots", plots)
        jnp.save("sampled", sampled)
    else:
        sampled = jnp.load("sampled.npy")
        plots = jnp.load("plots.npy")

    # Plot the samples and the denoising process from the trained or loaded model
    train_diffusion.plot_images(sampled.reshape(-1, 28, 28), "samples.pdf")
    train_diffusion.plot_images(plots[:, 0], "denoising_process.pdf")
