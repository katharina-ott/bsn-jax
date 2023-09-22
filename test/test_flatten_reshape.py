import jax.numpy as jnp

from model.param_util import get_info_from_list_of_tuples, flatten_pytree, unflatten_params


def test_flatten():
    params = [(jnp.array([2., 3.]),),(jnp.array([1., 2., 4.]), jnp.array([1., 2., 4.]))]
    params_flatten = flatten_pytree(params)
    assert len(params_flatten.shape) == 1


def test_flatten_reshape():
    params = [(jnp.array([2., 3.]),), (jnp.array([1., 2., 4.]), jnp.array([1., 2., 4.]))]
    lengths, shapes = get_info_from_list_of_tuples(params)
    params_flatten = flatten_pytree(params)
    params_reshape = unflatten_params(params_flatten, lengths, shapes)
    # Todo: check if params and params_reshape are the same

