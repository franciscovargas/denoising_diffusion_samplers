"""Simple sparse logistic regression target.
"""
import pickle
import jax

from jax.flatten_util import ravel_pytree

import jax.numpy as np
import numpy as onp

import numpyro
import numpyro.distributions as dists

from dds import data_paths

from os import path


def pad_with_const(x):
  extra = onp.ones((x.shape[0], 1))
  return onp.hstack([extra, x])


def standardize_and_pad(x):
  mean = onp.mean(x, axis=0)
  std = onp.std(x, axis=0)
  std[std == 0] = 1.
  x = (x - mean) / std
  return pad_with_const(x)


def load_data(name="sonar_full.pkl"):
  path_ = path.join(data_paths.data_path, name)
  with open(path_, mode="rb") as f:
    x, y = pickle.load(f)
  y = (y + 1) // 2
  x = standardize_and_pad(x)
  return x, y


def load_target(name="sonar_full.pkl"):
  """Loads target probalistic model.

  Args:
    name: filename for dataset

  Returns:
    tuple with function that evaluates log prob of the model and dim
  """

  def model(y_obs):
    w = numpyro.sample("weights", dists.Normal(np.zeros(dim), np.ones(dim)))
    logits = np.dot(x, w)
    with numpyro.plate("J", n_data):
      _ = numpyro.sample("y", dists.BernoulliLogits(logits), obs=y_obs)

  x, y_ = load_data(name)
  dim = x.shape[1]
  n_data = x.shape[0]
  model_args = (y_,)

  rng_key = jax.random.PRNGKey(1)
  model_param_info, potential_fn, _, _ = numpyro.infer.util.initialize_model(
      rng_key, model, model_args=model_args)
  params_flat, unflattener = ravel_pytree(model_param_info[0])

  log_prob_model = lambda z: -1. * potential_fn(unflattener(z))
  dim = params_flat.shape[0]
  # unflatten_and_constrain = lambda z: constrain_fn(unflattener(z))

  def log_prob(x):
    return jax.vmap(log_prob_model, in_axes=0)(x)

  return log_prob, dim
