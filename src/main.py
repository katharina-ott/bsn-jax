import itertools
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import scipy
import seaborn as seaborn
from jax import random, grad
from jax import vmap
from jax.example_libraries import optimizers
from matplotlib import pyplot as plt

from data import GenzContinuousDataSet1D
from model import init_network_params, apply_u_network, stein_operator
from options import Options
from util import PlOTTING_PATH


def get_info_from_list_of_tuples(params):
    """
    Obtains all the useful information of params to reconstruct the pytree from a flat array.
    :param params: pytree containing jnp.Array
    :return:
    """
    treedef = jax.tree_util.tree_structure(params)
    leaves = jax.tree_util.tree_leaves(params)
    shapes = [l.shape for l in leaves]
    return treedef, shapes


def flatten_pytree(params):
    """
    Flattens a pytree into a single jnp.Array
    :param params: pytree
    :return:
    """
    param_list = jax.tree_util.tree_leaves(params)
    flattened_params = jnp.concatenate([param.flatten() for param in param_list])
    return flattened_params


def unflatten_params(flattened_params, treedef, shapes):

    unflattened_list = []
    start_idx = 0

    for shape in shapes:
        size = np.prod(shape)
        end_idx = start_idx + size
        if end_idx <= len(flattened_params):
            unflattened_list.append(flattened_params[start_idx:end_idx].reshape(shape))
            start_idx = end_idx
        else:
            raise ValueError("Flattened array is smaller than expected.")

    unflattened_list = jax.tree_util.tree_unflatten(treedef, unflattened_list)
    return unflattened_list


def loss(params, x, y, score, apply_u_network):
    preds = batch_apply_stein(params, x, score, apply_u_network)
    loss = jnp.mean((preds - y) ** 2)
    return loss


def loss_param_array(params, x, y, score, apply_u_network, treedef, shapes):
    params = unflatten_params(params, treedef, shapes)
    return loss(params, x, y, score, apply_u_network)


batch_apply_stein = vmap(stein_operator, in_axes=(None, 0, 0, None))


def train_stein_network(x, score, y, params, num_epochs: int):
    """
    Optimizes the parameters using the adam optimizer
    :param x:
    :param score:
    :param y:
    :param params:
    :param num_epochs:
    :return:
    """
    opt_init, opt_update, get_params = optimizers.adam(step_size=opts.step_size)
    opt_state = opt_init(params)
    params = get_params(opt_state)
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


def train_stein_lbfgs(x, score, y, params, method):
    """
    Optimizes the parameters using any scipy.optimize method.
    :param x:
    :param score:
    :param y:
    :param params:
    :param method:
    :return:
    """
    params0 = params
    treedef, shapes = get_info_from_list_of_tuples(params0)
    params0 = flatten_pytree(params0)
    method = method
    start_time = time.time()
    fun = lambda p: loss_param_array(p, x, y, score, apply_u_network, treedef, shapes)
    jac = lambda p: grad(fun)(p)
    """
    WARNING: This is using scipy code - so no cuda capabilities, no jiting.
    I was unable to do parameters optimization with L-BFGS in jax.
    This needs to be looked into at a later point.
    """
    out = scipy.optimize.minimize(fun, x0=params0, method=method, jac=jac, tol=None, options=None)
    epoch_time = time.time() - start_time
    print(f"Run time optimization: {epoch_time}")
    params = unflatten_params(out.x, treedef, shapes)
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
    if opts.method == "adam":
        params = train_stein_network(x, score, y, params, opts.num_epochs)
    else:
        params = train_stein_lbfgs(x, score, y, params, opts.method)
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
                   method="L-BFGS-B",
                   num_epochs=1000,
                   layers_sizes=[[1, 32], [32, 1]],
                   n=10,
                   data_class=GenzContinuousDataSet1D
                   )

    run(opts, prng_key)
