import os
import pickle

import jax
# jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_disable_jit", True)

import jax.numpy as jnp
import pandas as pd
from jax import random

from metrics.metrics import get_energy_metric

from helpers import apply_function
from losses.loss_and_model import loss_model_bel, loss_model_no_train, loss_model_reparam
from problems.toy_problems import dm_toy_problem, double_well_toy_problem, double_well_toy_problem_opening, ou_toy_problem
from training.train_model import train_model
from problems.diffusion_model import get_problem_fn as get_problem_dm

import datetime
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

from flow_matching_jax.configs.fashion_mnist import get_config


print(f"Using Jax Device: {jax.devices()}")

log_dir = "logs/diffusion_model"
writer = tf.summary.create_file_writer(log_dir)

# Register hyperparameters in TensorBoard
# HP_OPTIMIZER = hp.HParam('loss_function', hp.Discrete(['bel', 'no_train']))
# with writer.as_default():
#     hp.hparams_config(
#         hparams=[HP_OPTIMIZER],
#         metrics=[hp.Metric("loss", display_name="Loss"), hp.Metric("energy", display_name="Energy")],
#     )


# dim = 1
ts = jnp.linspace(0, 1, 100, dtype=jnp.float32)

loss_models = [
    {"function": loss_model_bel, "args": ("first",)},
    # {"function": loss_model_bel, "args": ("average",)},
    # {"function": loss_model_bel, "args": ("first",)},
    # {"function": loss_model_bel, "args": ("last",)},
    # {"function": loss_model_reparam, "args": ()},
    # {"function": loss_model_no_train, "args": ()},
]

config = get_config()
workdir = "/home/ubuntu/WashingtonMain/conditioning-diffusions/flow_matching_jax/workdir/fashion_mnist"
dm_problem = get_problem_dm(config, workdir)
# problems = [double_well_toy_problem(3, D=dim), dm_toy_problem(D=dim),  ou_toy_problem(2, D=dim), ou_toy_problem(-2, D=dim), ou_toy_problem(0, D=dim)]
problems = [
    # {"function": double_well_toy_problem_opening, "args": (3,), "kwargs": {"D": dim}},
    # {"function": double_well_toy_problem, "args": (3,), "kwargs": {"D": dim}},
    # {"function": double_well_toy_problem, "args": (5,), "kwargs": {"D": dim}},
    # {"function": dm_toy_problem, "args": (), "kwargs": {"D": dim}},
    # {"function": ou_toy_problem, "args": (2,), "kwargs": {"D": dim}},
    # {"function": ou_toy_problem, "args": (-2,), "kwargs": {"D": dim}},
    # {"function": ou_toy_problem, "args": (0,), "kwargs": {"D": dim}},
    {"function": dm_problem, "args": (), "kwargs": {}},
]


N_iters = len(problems) * len(loss_models)
k = 0
data = []
data_full = []

config_template = {
    "seed": 1995,
    "loss_model": "NOT DEFINED",
    "problem": "NOT DEFINED",
    "N_batches": 5,
    "N_log": 1,
    "N_samples_eval": 10,
    "N_batch_size": 1,
    "ts": ts,
    "n_rngs": 1,
}

rng = random.PRNGKey(config_template["seed"])
rngs = random.split(rng, config_template["n_rngs"])

experiment_dir = f"temp"
os.makedirs(experiment_dir, exist_ok=True)  # Create the directory if it doesn't exist

l = 0
for rng in rngs:
    print(f"Rng number {k}")
    l += 1
    for i in range(len(problems)):
        for j in range(len(loss_models)):
            k += 1
            problem = apply_function(problems[i])
            loss_model = apply_function(loss_models[j])
            sde, metrics, y_obs, y_init_eval, problem_name = problem
            loss_function, nn_model, loss_name = loss_model

            config = config_template.copy()
            config["problem"] = problems[i]
            config["loss_model"] = loss_models[j]
            config["rng"] = rng

            print(f"{k}/{N_iters} Training {loss_name} on {problem_name}")

            # get a timestamp hhmmss-mmdd
            timestamp = datetime.datetime.now().strftime("%H%M%S-%m%d")
            run_dir = f"{log_dir}/run-{timestamp}-{k}"
            writer = tf.summary.create_file_writer(run_dir)

            # Pass the config arguments directly to the train_model function
            final_params, all_metrics, last_metrics = train_model(
                rng,
                ts,
                nn_model,
                metrics,
                y_obs,
                y_init_eval,
                sde,
                loss_function,
                writer,
                N_batches=config["N_batches"],
                N_batch_size=config["N_batch_size"],
                N_log=config["N_log"],
                N_samples_eval=config["N_samples_eval"],
            )

            print(last_metrics)

            # Save all_params, all_metrics, and config
            save_dir = os.path.join(experiment_dir, f"{problem_name}_{loss_name}")
            os.makedirs(save_dir, exist_ok=True)

            with open(os.path.join(save_dir, "final_params.pkl"), "wb") as f:
                pickle.dump(final_params, f)
            with open(os.path.join(save_dir, "all_metrics.pkl"), "wb") as f:
                pickle.dump(all_metrics, f)
            with open(os.path.join(save_dir, "config.pkl"), "wb") as f:
                pickle.dump(config, f)

            # Collect metrics
            data.append({"problem": problem_name, "loss": loss_name, **last_metrics})
            data_full.append({"problem": problem_name, "loss": loss_name, "metrics": all_metrics})

# Save data_full to a file for later retrieval
# data_full_path = os.path.join(experiment_dir, "data_full.pkl")
# with open(data_full_path, "wb") as f:
#     pickle.dump(data_full, f)

# Save summary results to a CSV
data_df = pd.DataFrame(data)
data_df.to_csv(os.path.join(experiment_dir, "results.csv"), index=False)
