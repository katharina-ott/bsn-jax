from typing import List, Tuple

import jax
import numpy as np
from jax import numpy as jnp
from jax._src.tree_util import PyTreeDef


def get_info_from_list_of_tuples(params: List[Tuple[jnp.ndarray, jnp.ndarray] | jnp.ndarray]) -> (PyTreeDef, Tuple):
    """
    Obtains all the useful information of params to reconstruct the pytree from a flat array.

    Parameters
    ----------
    params: List[Tuple[jnp.ndarray, jnp.ndarray] | jnp.ndarray]
        Parameters of the network in pytree form
    Returns
    ----------
    PyTreeDef
        A parameter describing the pytree of params
    Tuple
        A list of shapes describing the arrays in params
    """
    treedef = jax.tree_util.tree_structure(params)
    leaves = jax.tree_util.tree_leaves(params)
    shapes = [leaf.shape for leaf in leaves]
    return treedef, shapes


def flatten_pytree(params: List[Tuple[jnp.ndarray, jnp.ndarray] | jnp.ndarray]) -> jnp.ndarray:
    """
    Flattens a pytree into a single jnp.ndarray
    Parameters
    ----------
    params: List[Tuple[jnp.ndarray, jnp.ndarray] | jnp.ndarray]
        Parameters of the network in pytree form
    Returns
    ----------
    jnp.ndarray
        Flattened parameters
    """
    param_list = jax.tree_util.tree_leaves(params)
    flattened_params = jnp.concatenate([param.flatten() for param in param_list])
    return flattened_params


def unflatten_params(flattened_params: jnp.ndarray, treedef: PyTreeDef, shapes: Tuple) -> List[
    Tuple[jnp.ndarray, jnp.ndarray] | jnp.ndarray]:
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
