import jax
import jax.numpy as jnp
import numpy as onp
import optax
from jax import random
from tqdm import tqdm

from losses.update_step import update_step
from metrics.metrics import (
    amortized_mean_var_metric,
    endpoint_distance,
    endpoint_distance_amortized,
    kl_metrics,
    mean_var_metric,
)


def train_model(
    rng,
    ts,
    nn_model,
    true_control,
    y_obs,
    y_init_eval,
    sde,
    loss_function,
    N_batches=1_001,
    N_batch_size=512,
    N_log=100,
    N_samples_eval=2048,
):
    _t = 0.0
    D = y_init_eval.shape[0]
    nn_params = nn_model.init(rng, _t, y_init_eval, y_obs)

    optimizer = optax.adam(1e-3, b1=0.97)
    opt_state = optimizer.init(nn_params)

    losses = []

    initial_samples_eval = jnp.ones((N_samples_eval, D)) * 1

    rng, eval_rng = random.split(rng)
    all_metrics = []
    metric_true_drift = []

    losses_history = []

    grad_norms = []
    grad_norms_variance = []

    # LOSS FUNCTIONS

    # loss_function = lambda *args: get_loss_nik_consistency(*args, mode="soc")
    # loss_function = get_loss
    # loss_function = lambda *args: get_loss_nik(*args, mode="last")#, integration_mode="first_J")

    # sde = bm
    # true_control = brownian_bridge(y_obs_val)

    # sde, true_control = double_well_problem(3, y_obs_val)
    # sde, true_control = ou_problem(5, y_obs_val)
    # sde, true_control = sin_problem(3, y_obs_val)
    # sde, _ = dm_problem(0.01, 2, y_obs_val)

    rng, srng = random.split(rng)
    if true_control is not None:
        mv_metric = mean_var_metric(srng, sde, ts, true_control, initial_samples_eval)
        mv_amortized = amortized_mean_var_metric(rng, sde, ts, true_control, initial_samples_eval)
        endpoint_amortized = endpoint_distance_amortized(rng, sde, ts, true_control, initial_samples_eval)
        endpoint_rare_event = endpoint_distance(rng, sde, ts, true_control, initial_samples_eval)
        metrics = {
            "kl": kl_metrics,
            "mean_var_rare_event": mv_metric,
            "mean_var_amortized": mv_amortized,
            "endpoint_distance": endpoint_amortized,
            "endpoint_distance_rare_event": endpoint_rare_event,
        }
    else:
        metrics = {}
    
    # best_params = nn_params.copy()
    # min_metric = jnp.inf

    x0s = jnp.tile(y_init_eval, (N_batch_size, 1))


    # main_metric = "mean_var_rare_event"

    best_epoch = -1
    for i in tqdm(range(N_batches), desc="Model Training"):
        rng, srng = random.split(rng)
        # x0s = random.uniform(srng, (N_batch_size, D)) * 3 - 1.5
        # x0s = initial_sampler(srng, N_batch_size)
        rng, srng = random.split(rng)
        val, nn_params, opt_state, grad_norm = update_step(
            srng, loss_function, sde, nn_model, nn_params, ts, x0s, opt_state, optimizer, y_obs, grad_clip=1
        )
        losses.append(val)
        grad_norms.append(grad_norm)
        if (i + 1) % N_log == 0:
            # control = lambda t, x, y: nn_model.apply(nn_params, t, x, y)
            loss_avg = onp.mean(losses)
            # print(f"\tLoss: {loss_avg}")
            losses_history.append(loss_avg)
            grad_var = onp.var(grad_norms)
            grad_norms_variance.append(grad_var)
            grad_norms = []

            # print(f"Grad norm is {grad_norm}")
            # weight_norm = jnp.sqrt(
            #     jax.tree_util.tree_reduce(lambda acc, g: acc + jnp.sum(g**2), nn_params, initializer=0.0)
            # )
            # print(f"Weight norm is {weight_norm}")
            losses = []

            # If Likelihood is known (not for true bridges)
            # m = metric(eval_rng, sde, ts, control, initial_samples_eval, likelihood, y)
            # print("\t Importance Sampling Metric: ", m)
            # metric_importance_sampling.append(m)

            # if true drift is known
            # m2 = kl_metrics(eval_rng, sde, ts, nn_model, nn_params, initial_samples_eval, true_control, y_obs_arr, "generated")

            metrics_log = {"grad_variance": grad_var}
            for metric_name, metric in metrics.items():
                metrics_log[metric_name] = metric(
                    eval_rng, sde, ts, nn_model, nn_params, initial_samples_eval, true_control, y_obs
                )
            # kl_m = kl_metrics(eval_rng, sde, ts, nn_model, nn_params, initial_samples_eval, true_control, y_obs, "true")
            # kl_m_gen = kl_metrics(eval_rng, sde, ts, nn_model, nn_params, initial_samples_eval, true_control, y_obs, "generated")
            # mv_m = mv_metric(eval_rng, sde, ts, nn_model, nn_params, initial_samples_eval, true_control, y_obs)
            # all_metrics.append({"kl_true": kl_m, "mean_var": mv_m, "kl_gen": kl_m_gen, "grad_variance": grad_var})
            all_metrics.append(metrics_log)

            # if metrics_log[main_metric] < min_metric:
            #     best_params = nn_params.copy()
            #     min_metric = metrics_log[main_metric]
            #     best_epoch = i + 1

    all_metrics = {key: [d[key] for d in all_metrics] for key in all_metrics[0]}
    last_metrics = {}
    for metric_name, metric in metrics.items():
        # best = onp.min(all_metrics[metric_name])
        last_metrics[metric_name] = all_metrics[metric_name][-1]

    average_grad_var = onp.mean(all_metrics["grad_variance"][1:])
    last_metrics["avg_grad_var"] = average_grad_var
    # best_metrics["best_epoch"] = best_epoch
    # all_metrics["best_epoch"] = best_epoch

    # best_metrics = {"kl": best_kl, "mean_var": best_mv, "grad_variance": average_grad_var, "best_epoch": best_epoch}

    # BEST METRICS ARE ACTUALLY JUST LAST METRICS NOW

    return nn_params, all_metrics, last_metrics
