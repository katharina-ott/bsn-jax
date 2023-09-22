import time
from typing import List, Tuple

import scipy
from jax import numpy as jnp, grad
from jax.example_libraries import optimizers

from model.loss import loss, loss_param_array
from model.model import apply_u_network
from model.param_util import get_info_from_list_of_tuples, flatten_pytree, unflatten_params


def train_stein_network(x: jnp.ndarray, score: jnp.ndarray, y: jnp.ndarray,
                        params: List[Tuple[jnp.ndarray, jnp.ndarray] | jnp.ndarray], num_epochs: int,
                        step_size: float) -> List[Tuple[jnp.ndarray, jnp.ndarray] | jnp.ndarray]:
    """
    Optimizes the parameters using the adam optimizer

    Parameters
    ----------
    x: jnp.ndarray
    score: jnp.ndarray
    y: jnp.ndarray
    params: List[Tuple[jnp.ndarray, jnp.ndarray] | jnp.ndarray]
        Parameters of the network in pytree form
    num_epochs: int

    Returns
    ----------
    List[Tuple[jnp.ndarray, jnp.ndarray] | jnp.ndarray]
        Trained parameters of the network
    """
    opt_init, opt_update, get_params = optimizers.adam(step_size=step_size)
    opt_state = opt_init(params)
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


def train_stein_lbfgs(x: jnp.ndarray, score: jnp.ndarray, y: jnp.ndarray,
                      params: List[Tuple[jnp.ndarray, jnp.ndarray] | jnp.ndarray], method: str) -> List[
    Tuple[jnp.ndarray, jnp.ndarray] | jnp.ndarray]:
    """
    Optimizes the parameters using any `scipy.optimize` method.
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
    start_time = time.time()
    fun = lambda p: loss_param_array(p, x, y, score, apply_u_network, treedef, shapes)
    jac = lambda p: grad(fun)(p)
    """
    WARNING: This is using scipy code - so no cuda capabilities, no jitting.
    I was unable to do parameters optimization with L-BFGS in jax.
    This needs to be looked into at a later point.
    """
    out = scipy.optimize.minimize(fun, x0=params0, method=method, jac=jac, tol=1.e-15,
                                  options={"maxiter": 2000})
    epoch_time = time.time() - start_time
    print(f"Run time optimization: {epoch_time}")
    print(f"Optimziations has converged: {out.success}")
    params = unflatten_params(out.x, treedef, shapes)
    return params
