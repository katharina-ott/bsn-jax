import itertools
import os

import jax
import matplotlib.pyplot as plt
import numpy as np
import seaborn

from data import GenzContinuousDataSet1D
from main import run
from options import Options
from util import PlOTTING_PATH


def plot_relative_error():
    relative_network_error = np.abs(network_estimate_list - true_value) / true_value
    relative_mcmc_error = np.abs(mcmc_estimate_list - true_value) / true_value
    seaborn.set_theme(style="whitegrid")
    palette = itertools.cycle(seaborn.color_palette())
    fig, ax = plt.subplots()
    ax.loglog(n_list, relative_network_error, "v-", color=next(palette), label="BSN")
    ax.loglog(n_list, relative_mcmc_error, "o-", color=next(palette), label="MCMC")
    ax.set_xlabel("#Data Points")
    ax.set_ylabel("Relative Error")
    ax.legend()
    fig.savefig(os.path.join(PlOTTING_PATH, "relative_error.png"), dpi=500)
    plt.clf()


if __name__ == "__main__":
    prng_key = jax.random.PRNGKey(0)
    opts = Options(step_size=0.01,
                   method="L-BFGS-B",
                   num_epochs=500,
                   layers_sizes=[[1, 32], [32, 32], [32, 32], [32, 1]],
                   n=5,
                   data_class=GenzContinuousDataSet1D
                   )

    n_list = np.array([5, 10, 20, 40, 80, 160])
    network_estimate_list = []
    mcmc_estimate_list = []
    for n in n_list:
        opts.n = n
        network_estimate, mcmc_estimate, true_value = run(opts, prng_key)
        network_estimate_list.append(network_estimate)
        mcmc_estimate_list.append(mcmc_estimate)

    network_estimate_list = np.array(network_estimate_list)
    mcmc_estimate_list = np.array(mcmc_estimate_list)

    plot_relative_error()
