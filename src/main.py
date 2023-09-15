import itertools
import os
import time

import jax
import jax.numpy as jnp
import seaborn as seaborn
from jax import random, grad
from jax import vmap
from jax.example_libraries import optimizers
from matplotlib import pyplot as plt

from data import GenzContinuousDataSet1D, GenzDiscontinuousDataSet1D


def random_layer_params(m, n, key, scale=1e-2):
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))


def random_theta0(key, scale=1e-2):
    return scale * random.normal(key, (1,))


def init_network_params(sizes, key):
    keys = random.split(key, 2)
    l = []
    for m, n, k in zip(sizes[:][0], sizes[:][1], keys):
        l.append(random_layer_params(m, n, k))
    l.append(random_theta0(key))
    return l


def activation(x):
    return jnp.tanh(x)


def apply_u_network(params, x):
    activations = x
    for w, b in params[:-1]:
        outputs = jnp.dot(w, activations) + b
        activations = activation(outputs)
    final_w, final_b = params[-1]
    logits = jnp.dot(final_w, activations) + final_b
    return logits


def calc_div_u(params, x, apply_u_network):
    r"""
    Computes
     .. math::
        \nabla u = \sum_{i=1}^d \frac{du_i}{x_i}
    where
     .. math::
         \begin{array}{ll} \\
            x \in \mathbb{R}^{1\times d}\\
            u(x) \in \mathbb{R}^{1\times d}\\
         \end{array}
    :param params: network parameters
    :param x: single element of dataset
    :param apply_u_network: the network u
    :return: divergence of u - 1-dimensional
    """
    u_func = lambda x: apply_u_network(params, x)
    dudx = jax.jacrev(u_func)(x).reshape(x.shape[0], x.shape[0])
    dudx = jnp.diag(dudx)
    divu = jnp.sum(dudx)
    return divu


def stein_operator(params, x, score, apply_u_network):
    r"""
    Applies the Stein operator to the network u:
     .. math::
        S[u] = \left(\nabla \log \pi(x)\right)^T u(x) + \nabla u(x)

    :param params: parameters of the network
    :param x: single element of neural network
    :param score: score function evaluated at x
    :param apply_u_network: the network u
    :return:
    """
    params, theta0 = params[:-1], params[-1]
    u = apply_u_network(params, x)
    stein = jnp.dot(score, u) + calc_div_u(params, x, apply_u_network) + theta0
    return stein


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
    fig.savefig(os.path.join("network_fit.png"), dpi=500)
    print("==========================================")
    if true_integral is not None:
        print(f"True integral value: {true_integral.item()}")
    network_estimate = params[-1].item()
    print(f"Network computed value: {network_estimate}")
    if mc_value is not None:
        print(f"MC computed value: {mc_value}")
    print("==========================================")
    return network_estimate


def run():
    params = init_network_params(layer_sizes, prng_key)
    dataset = data_class()
    x, score, y, x_test, score_test = dataset.return_data_set(n_targets)
    opt_init, opt_update, get_params = optimizers.adam(step_size=step_size)
    opt_state = opt_init(params)
    params = train_stein_network(x, score, y, opt_state, opt_update, get_params, num_epochs)
    true_integral = dataset.true_integration_value()
    mc_estimate = jnp.mean(y)
    network_estimate = evaluate_model(params, x, y, x_test, score_test,
                                      true_integral=true_integral,
                                      mc_value=mc_estimate
                                      )
    return network_estimate, mc_estimate, true_integral


if __name__ == "__main__":
    step_size = 0.01
    num_epochs = 1000
    layer_sizes = [[1, 32], [32, 1]]
    n_targets = 20
    data_class = GenzDiscontinuousDataSet1D
    prng_key = random.PRNGKey(0)

    run()
