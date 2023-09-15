import jax
from jax import random, numpy as jnp


def random_layer_params(m, n, key, scale=1e-2):
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))


def random_theta0(key, scale=1e-2):
    return scale * random.normal(key, (1,))


def init_network_params(sizes, key):
    keys = random.split(key, len(sizes))
    l = []
    for (m, n), k in zip(sizes[:], keys):
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
