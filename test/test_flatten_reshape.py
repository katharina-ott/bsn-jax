import jax.numpy as jnp

from bsn_jax.model.param_util import get_info_from_list_of_tuples, flatten_pytree, unflatten_params


def test_flatten():
    params = [(jnp.array([2.0, 3.0]),), (jnp.array([1.0, 2.0, 4.0]), jnp.array([1.0, 2.0, 4.0]))]
    params_flatten = flatten_pytree(params)
    assert len(params_flatten.shape) == 1


def test_flatten_reshape():
    params = [(jnp.array([2.0, 3.0]),), (jnp.array([1.0, 2.0, 4.0]), jnp.array([1.0, 2.0, 4.0]))]
    lengths, shapes = get_info_from_list_of_tuples(params)
    params_flatten = flatten_pytree(params)
    params_reshape = unflatten_params(params_flatten, lengths, shapes)
    # Todo: check if params and params_reshape are the same
