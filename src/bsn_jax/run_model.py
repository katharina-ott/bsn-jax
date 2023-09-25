from typing import Tuple

from jax import numpy as jnp
from jax.random import PRNGKeyArray

from bsn_jax.evaluate import evaluate_model
from bsn_jax.model.model import init_network_params
from bsn_jax.options import Options
from bsn_jax.train_model_loop import train_stein_network, train_stein_lbfgs


def run(opts: Options, prng_key: PRNGKeyArray) -> Tuple[float, float, float]:
    """
    Training method for a BSN. `opts` defines the settings used for training.

    Parameters
    ----------
    opts: Options
        Defines the settings, e.g., learning rate, choice of optimizer, network architecture.
    prng_key: PRNGKeyArray
        Random seed used for training

    Returns
    -------
    Tuple[float, float, Union[None,float]]
        Estimates obtained for the integral.
    """
    params = init_network_params(opts.layer_sizes, prng_key)
    dataset = opts.data_class()
    x, score, y, x_test, score_test = dataset.return_data_set(opts.n)
    if opts.method == "adam":
        params = train_stein_network(x, score, y, params, opts.num_epochs, opts.step_size)
    else:
        params = train_stein_lbfgs(x, score, y, params, opts.method)
    true_integral = dataset.true_integration_value()
    mc_estimate = jnp.mean(y).item()
    network_estimate = evaluate_model(
        params, x, y, x_test, score_test, true_integral=true_integral, mc_value=mc_estimate
    )
    return network_estimate, mc_estimate, true_integral
