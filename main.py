import time

import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax.example_libraries import optimizers
import matplotlib.pyplot as plt


def random_layer_params(m, n, key, scale=1e-2):
  w_key, b_key = random.split(key)
  return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))

# Initialize all layers for a fully-connected neural network with sizes "sizes"
def init_network_params(sizes, key):
  keys = random.split(key, 2)
  l = []
  for m, n, k in zip(sizes[:][0], sizes[:][1], keys):
    print(m, n, k)
    l.append(random_layer_params(m, n, k))
  return l


def relu(x):
  return jnp.maximum(0, x)


def apply_network(params, x):
  # per-example predictions
  activations = x
  for w, b in params[:-1]:
    outputs = jnp.dot(w, activations) + b
    activations = relu(outputs)

  final_w, final_b = params[-1]
  logits = jnp.dot(final_w, activations) + final_b
  return logits


def loss(params, x, y):
  preds = batch_apply_network(params, x)
  loss = jnp.mean((preds-y)**2)
  return loss

step_size = 0.01
num_epochs = 1000
batch_size = 1
n_targets = 10
layer_sizes = [[1, 32], [32, 1]]
params = init_network_params(layer_sizes, random.PRNGKey(0))
batch_apply_network = vmap(apply_network, in_axes=(None, 0))
x = jnp.linspace(-1, 1, 10)[:, None]
y = jnp.exp(- x**2)
y = y/jnp.max(y)
out = batch_apply_network(params, x)
opt_init, opt_update, get_params = optimizers.adam(step_size=0.01)
opt_state = opt_init(params)
for epoch in range(num_epochs):
  start_time = time.time()
  params = get_params(opt_state)
  g = grad(loss)(params, x, y)
  opt_state = opt_update(epoch, g, opt_state)
  epoch_time = time.time() - start_time

  current_loss = loss(params, x, y)
  if epoch%10 == 0:
    print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
    print("Training set accuracy {}".format(current_loss))

#
params = get_params(opt_state)
out = batch_apply_network(params, x)
plt.plot(x.flatten(), y.flatten())
plt.plot(x.flatten(), out.flatten())
plt.show()




# if __name__ == '__main__':
