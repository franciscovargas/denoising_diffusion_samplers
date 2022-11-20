"""Wrapper around aft_densities in case customisation is needed.
"""
import abc
import functools
from typing import Any, Optional

import chex
import haiku as hk

import inference_gym.using_jax as gym


import jax
import jax.numpy as jnp

import numpy as np


from dds.targets import cp_utils
from dds.targets import lr_sonar


from annealed_flow_transport.densities import AutoEncoderLikelihood
from annealed_flow_transport.densities import FunnelDistribution
from annealed_flow_transport.densities import LogDensity
from annealed_flow_transport.densities import MultivariateNormalDistribution


# TypeDefs
NpArray = np.ndarray
Array = jnp.ndarray
ConfigDict = Any

# imported likes
__LIKES__ = [
    AutoEncoderLikelihood, FunnelDistribution, MultivariateNormalDistribution
]


class LogGaussianCoxPines(LogDensity):
  """Log Gaussian Cox process posterior in 2D for pine saplings data.

  This follows Heng et al 2020 https://arxiv.org/abs/1708.08396 .

  config.file_path should point to a csv file of num_points columns
  and 2 rows containg the Finnish pines data.

  config.use_whitened is a boolean specifying whether or not to use a
  reparameterization in terms of the Cholesky decomposition of the prior.
  See Section G.4 of https://arxiv.org/abs/2102.07501 for more detail.
  The experiments in the paper have this set to False.

  num_dim should be the square of the lattice sites per dimension.
  So for a 40 x 40 grid num_dim should be 1600.
  """

  def __init__(self,
               config: ConfigDict,
               num_dim: int = 1600):
    super().__init__(config, num_dim)

    # Discretization is as in Controlled Sequential Monte Carlo
    # by Heng et al 2017 https://arxiv.org/abs/1708.08396
    self._num_latents = num_dim
    self._num_grid_per_dim = int(np.sqrt(num_dim))

    bin_counts = jnp.array(
        cp_utils.get_bin_counts(self.get_pines_points(config.file_path),
                                self._num_grid_per_dim))

    self._flat_bin_counts = jnp.reshape(bin_counts, (self._num_latents))

    # This normalizes by the number of elements in the grid
    self._poisson_a = 1./self._num_latents
    # Parameters for LGCP are as estimated in Moller et al, 1998
    # "Log Gaussian Cox processes" and are also used in Heng et al.

    self._signal_variance = 1.91
    self._beta = 1./33

    self._bin_vals = cp_utils.get_bin_vals(self._num_grid_per_dim)

    def short_kernel_func(x, y):
      return cp_utils.kernel_func(x, y, self._signal_variance,
                                  self._num_grid_per_dim, self._beta)

    self._gram_matrix = cp_utils.gram(short_kernel_func, self._bin_vals)
    self._cholesky_gram = jnp.linalg.cholesky(self._gram_matrix)
    self._white_gaussian_log_normalizer = -0.5 * self._num_latents * jnp.log(
        2. * jnp.pi)

    half_log_det_gram = jnp.sum(jnp.log(jnp.abs(jnp.diag(self._cholesky_gram))))
    self._unwhitened_gaussian_log_normalizer = -0.5 * self._num_latents * jnp.log(
        2. * jnp.pi) - half_log_det_gram
    # The mean function is a constant with value mu_zero.
    self._mu_zero = jnp.log(126.) - 0.5*self._signal_variance

    if self._config.use_whitened:
      self._posterior_log_density = self.whitened_posterior_log_density
    else:
      self._posterior_log_density = self.unwhitened_posterior_log_density

  def  _check_constructor_inputs(self, config: ConfigDict, num_dim: int):
    expected_members_types = [("use_whitened", bool)]
    self._check_members_types(config, expected_members_types)
    num_grid_per_dim = int(np.sqrt(num_dim))
    if num_grid_per_dim * num_grid_per_dim != num_dim:
      msg = ("num_dim needs to be a square number for LogGaussianCoxPines "
             "density.")
      raise ValueError(msg)

    if not config.file_path:
      msg = "Please specify a path in config for the Finnish pines data csv."
      raise ValueError(msg)

  def get_pines_points(self, file_path):
    """Get the pines data points."""
    with open(file_path, mode="rt") as input_file:
    # with open(file_path, "rt") as input_file:
      b = np.genfromtxt(input_file, delimiter=",")
    return b

  def whitened_posterior_log_density(self, white: Array) -> Array:
    quadratic_term = -0.5 * jnp.sum(white**2)
    prior_log_density = self._white_gaussian_log_normalizer + quadratic_term
    latent_function = cp_utils.get_latents_from_white(white, self._mu_zero,
                                                      self._cholesky_gram)
    log_likelihood = cp_utils.poisson_process_log_likelihood(
        latent_function, self._poisson_a, self._flat_bin_counts)
    return prior_log_density + log_likelihood

  def unwhitened_posterior_log_density(self, latents: Array) -> Array:
    white = cp_utils.get_white_from_latents(latents, self._mu_zero,
                                            self._cholesky_gram)
    prior_log_density = -0.5 * jnp.sum(
        white * white) + self._unwhitened_gaussian_log_normalizer
    log_likelihood = cp_utils.poisson_process_log_likelihood(
        latents, self._poisson_a, self._flat_bin_counts)
    return prior_log_density + log_likelihood

  def evaluate_log_density(self, x: Array) -> Array:
    # import pdb; pdb.set_trace()
    if len(x.shape) == 1:
      return self._posterior_log_density(x)
    else:
      return jax.vmap(self._posterior_log_density)(x)


class LogReg(LogDensity):
  """This class implements the BrownianMissingMiddle from inf gym.

  We wrap it for compatibility purposes.
  """

  def __init__(self,
               config: ConfigDict,
               num_dim: int = 61):
    super().__init__(config, num_dim)
    self.dataset_name = config.dataset_name
    self.target, _ = lr_sonar.load_target(self.dataset_name)
    self.dim = num_dim

  def _check_constructor_inputs(self, config: ConfigDict, num_dim: int):
    pass

  def evaluate_log_density(self, x: Array) -> Array:
    return self.target(x)


class BrownianMissingMiddleScales(LogDensity):
  """This class implements the BrownianMissingMiddle from inf gym.

  We wrap it for compatibility purposes.
  """

  def __init__(self,
               config: ConfigDict,
               num_dim: int = 32):
    super().__init__(config, num_dim)
    self.cls = gym.targets.BrownianMotionUnknownScalesMissingMiddleObservations
    self.gym_target = gym.targets.VectorModel(
        self.cls(), flatten_sample_transformations=True)

  def _check_constructor_inputs(self, config: ConfigDict, num_dim: int):
    pass

  def _posterior_log_density(self, x: Array) -> Array:

    y = self.gym_target.default_event_space_bijector(x)

    lnp = self.gym_target.unnormalized_log_prob(y)
    jac = self.gym_target.default_event_space_bijector.forward_log_det_jacobian(
        x, event_ndims=1)
    return lnp + jac

  def evaluate_log_density(self, x: Array) -> Array:
    if len(x.shape) == 1:
      return self._posterior_log_density(x)
    else:
      return jax.vmap(self._posterior_log_density)(x)


class LorenzBridge(BrownianMissingMiddleScales):
  """This class implements the Lorentz time series model from inf gym.

  We wrap it for compatibility purposes.
  """

  def __init__(self,
               config: ConfigDict,
               num_dim: int = 90):
    super().__init__(config, num_dim)
    self.cls = gym.targets.ConvectionLorenzBridge
    self.gym_target = gym.targets.VectorModel(
        self.cls(), flatten_sample_transformations=True)
