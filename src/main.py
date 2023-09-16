import itertools
import os
import time

import jax.numpy as jnp
import seaborn as seaborn
from jax import random, grad
from jax import vmap
from jax.example_libraries import optimizers
from matplotlib import pyplot as plt

from data import GenzContinuousDataSet1D, GenzProductpeakDataSet1D
from model import init_network_params, apply_u_network, stein_operator
from options import Options
from util import PlOTTING_PATH


def loss(params, x, y, score, apply_u_network):
    preds = batch_apply_stein(params, x, score, apply_u_network)
    loss = jnp.mean((preds - y) ** 2)
    return loss


batch_apply_stein = vmap(stein_operator, in_axes=(None, 0, 0, None))


def train_stein_network(x, score, y, opt_state, opt_update, get_params, num_epochs: int):
    for epoch in range(num_epochs):
        start_time = time.time()
        params = get_params(opt_state)
        g = grad(loss)(params, x, y, score, apply_u_network)
        opt_state = opt_update(epoch, g, opt_state)
        epoch_time = time.time() - start_time

        current_loss = loss(params, x, y, score, apply_u_network)
        if epoch % 10 == 0:
            print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
            print("Training set accuracy {}".format(current_loss))
            print(f"Theta_0: {params[-1]}")
    return params


def evaluate_model(params, x, y, x_test, score_test, true_integral=None, mc_value=None):
    out = batch_apply_stein(params, x_test, score_test, apply_u_network)
    seaborn.set_theme(style="whitegrid")
    palette = itertools.cycle(seaborn.color_palette())
    fig, ax = plt.subplots()
    ax.scatter(x.flatten(), y.flatten(), color=next(palette), label="Training data")
    ax.plot(x_test.flatten(), out.flatten(), color=next(palette), label="Network fit")
    ax.legend()
    fig.savefig(os.path.join(PlOTTING_PATH, "network_fit.png"), dpi=500)
    plt.clf()
    print("==========================================")
    if true_integral is not None:
        print(f"True integral value: {true_integral.item()}")
    network_estimate = params[-1].item()
    print(f"Network computed value: {network_estimate}")
    if mc_value is not None:
        print(f"MC computed value: {mc_value}")
    print("==========================================")
    return network_estimate


def run(opts: Options, prng_key):
    params = init_network_params(opts.layer_sizes, prng_key)
    dataset = opts.data_class()
    x, score, y, x_test, score_test = dataset.return_data_set(opts.n)
    opt_init, opt_update, get_params = optimizers.adam(step_size=opts.step_size)
    opt_state = opt_init(params)
    params = train_stein_network(x, score, y, opt_state, opt_update, get_params, opts.num_epochs)
    true_integral = dataset.true_integration_value()
    mc_estimate = jnp.mean(y)
    network_estimate = evaluate_model(params, x, y, x_test, score_test,
                                      true_integral=true_integral,
                                      mc_value=mc_estimate
                                      )
    return network_estimate, mc_estimate, true_integral


if __name__ == "__main__":
    prng_key = random.PRNGKey(0)
    opts = Options(step_size=0.01,
                   num_epochs=1000,
                   layers_sizes=[[1, 32], [32, 32], [32, 32], [32, 1]],
                   n=32,
                   data_class=GenzContinuousDataSet1D
                   )

    run(opts)
