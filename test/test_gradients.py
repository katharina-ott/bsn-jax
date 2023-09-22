import jax.numpy as jnp

from bsn_jax.model import calc_div_u


def test_gradient():
    def apply_network(params, x):
        return params * x**2
    x = jnp.array([[5., 7, 9.], [2., 3., 5.]]).T
    params = jnp.array([[1., 3.]])

    u = calc_div_u(params, x, apply_network)
    manual_result = []
    for x_i in x:
        manual_result.append(jnp.sum(2*params*x_i))
    manual_result = jnp.array(manual_result)
    assert jnp.array_equal(u, manual_result)
