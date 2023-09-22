from typing import List, Tuple, Callable

from jax import numpy as jnp
from jax._src.tree_util import PyTreeDef

from model.model import batch_apply_stein
from model.param_util import unflatten_params


def loss(params: List[Tuple[jnp.ndarray, jnp.ndarray] | jnp.ndarray], x: jnp.ndarray, y: jnp.ndarray,
         score: jnp.ndarray, apply_u_network: Callable) -> jnp.ndarray:
    preds = batch_apply_stein(params, x, score, apply_u_network)
    loss = jnp.mean((preds - y) ** 2)
    return loss


def loss_param_array(params: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray, score: jnp.ndarray, apply_u_network: Callable,
                     treedef: PyTreeDef, shapes: Tuple) -> jnp.ndarray:
    params = unflatten_params(params, treedef, shapes)
    return loss(params, x, y, score, apply_u_network)
