import itertools
import os
from typing import List, Tuple

import seaborn as seaborn
from jax import numpy as jnp
from matplotlib import pyplot as plt

from bsn_jax.model.model import batch_apply_stein, apply_u_network
from bsn_jax.util import PlOTTING_PATH


def evaluate_model(
    params: List[Tuple[jnp.ndarray, jnp.ndarray] | jnp.ndarray],
    x: jnp.ndarray,
    y: jnp.ndarray,
    x_test: jnp.ndarray,
    score_test: jnp.ndarray,
    true_integral: float = None,
    mc_value: float = None,
):
    out = batch_apply_stein(params, x_test, score_test, apply_u_network)
    seaborn.set_theme(style="whitegrid")
    palette = itertools.cycle(seaborn.color_palette())
    fig, ax = plt.subplots()
    ax.scatter(x.flatten(), y.flatten(), color=next(palette), label="Training data")
    ax.plot(x_test.flatten(), out.flatten(), color=next(palette), label="Network fit")
    ax.legend()
    fig.savefig(os.path.join(PlOTTING_PATH, "network_fit.png"), dpi=500)
    plt.close(fig)
    plt.clf()
    print("==========================================")
    if true_integral is not None:
        print(f"True integral value: {true_integral}")
    network_estimate = params[-1].item()
    print(f"Network computed value: {network_estimate}")
    if mc_value is not None:
        print(f"MC computed value: {mc_value}")
    print("==========================================")
    return network_estimate
