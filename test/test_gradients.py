import jax.numpy as jnp

from bsn_jax.model.model import calc_div_u


def test_gradient():
    """
    Tests computation of the gradient.
    Note that calc_div_u can only handle single datapoints (as it is used within vmap).
    """

    def apply_network(params, x):
        return 0.5*params * x**2

    x = jnp.array([5.0, 7, 9.0]).T
    params = jnp.array([1.0, 3.0, 4.])

    u = calc_div_u(params, x, apply_network)
    manual_result = jnp.sum(params*x)
    assert jnp.array_equal(u, manual_result)
