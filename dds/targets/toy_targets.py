"""Distributions and datasets for sampler debugging.
"""
import distrax

import jax
import jax.numpy as jnp

from jax.scipy.special import logsumexp
from jax.scipy.stats import multivariate_normal
from jax.scipy.stats import norm

import numpy as np


def funnel(d=10, sig=3, clip_y=11):
  """Funnel distribution for testing. Returns energy and sample functions."""

  def neg_energy(x):
    def unbatched(x):
      v = x[0]
      log_density_v = norm.logpdf(v,
                                  loc=0.,
                                  scale=3.)
      variance_other = jnp.exp(v)
      other_dim = d - 1
      cov_other = jnp.eye(other_dim) * variance_other
      mean_other = jnp.zeros(other_dim)
      log_density_other = multivariate_normal.logpdf(x[1:],
                                                     mean=mean_other,
                                                     cov=cov_other)
      return log_density_v + log_density_other
    output = jax.vmap(unbatched)(x)
    return output

  def sample_data(n_samples):
    # sample from Nd funnel distribution
    y = (sig * jnp.array(np.random.randn(n_samples, 1))).clip(-clip_y, clip_y)
    x = jnp.array(np.random.randn(n_samples, d - 1)) * jnp.exp(-y / 2)
    return jnp.concatenate((y, x), axis=1)

  return neg_energy, sample_data


def simple_gaussian(d=2, sigma=1):
  """Wrapper method for simple Gaussian test distribution.

  Args:
    d: dim of N(0,sigma^2)
    sigma: scale/std of gaussian dist

  Returns:
    Tuple with log density, None and plotting func
  """

  dist = distrax.MultivariateNormalDiag(
      np.zeros(d), sigma * np.ones(d))

  log_p_pure = dist.log_prob

  def plot_distribution(ax):
    rngx = np.linspace(-2, 2, 100)
    rngy = np.linspace(-2.5, 2.5, 100)
    xx, yy = np.meshgrid(rngx, rngy)
    coords = np.dstack([xx, yy])
    coords = coords.reshape(-1, 2)
    log_ps = log_p_pure(coords)
    z = log_ps.reshape(100, 100)
    z = np.exp(z)
    ax.contourf(xx, yy, z, levels=50)

  return log_p_pure, None, plot_distribution


def far_gaussian(d=2, sigma=1, mean=6.0):
  """Wrapper method for simple Gaussian test distribution.

  Args:
    d: dim of N(0,sigma^2).
    sigma: scale/std of gaussian dist.
    mean: mean of gaussian.

  Returns:
    Tuple with log density, None and plotting func
  """

  dist = distrax.MultivariateNormalDiag(
      np.zeros(d) + mean, sigma * np.ones(d))

  log_p_pure = dist.log_prob

  def plot_distribution(ax, xrng=None, yrng=None, cmap=None):
    xrng = [-2, 2] if xrng is None else xrng
    yrng = [-2, 2] if yrng is None else yrng

    rngx = np.linspace(xrng[0], xrng[1], 100)
    rngy = np.linspace(yrng[0], yrng[1], 100)
    xx, yy = np.meshgrid(rngx, rngy)
    coords = np.dstack([xx, yy])
    coords = coords.reshape(-1, 2)
    log_ps = log_p_pure(coords)
    z = log_ps.reshape(100, 100)
    z = np.exp(z)
    ax.contourf(xx, yy, z, levels=50, cmap=cmap)

  return log_p_pure, None, plot_distribution


def mixture_well():
  """Wrapper method for well of mixtures target.

  Returns:
    tuple log density for mixture well and None
  """

  def euclidean_distance_einsum(x, y):
    """Efficiently calculates the euclidean distance between vectors in two mats.


    Args:
      x: first matrix (nxd)
      y: second matrix (mxd)

    Returns:
      pairwise distance matrix (nxm)
    """
    xx = jnp.einsum('ij,ij->i', x, x)[:, jnp.newaxis]
    yy = jnp.einsum('ij,ij->i', y, y)
    xy = 2 * jnp.dot(x, y.T)
    out = xx + yy - xy

    return out

  def log_p_pure(x):
    """Gaussian mixture density on well like structure.

    Args:
      x: vectors over which to evaluate the density

    Returns:
      nx1 vector containing density evaluations
    """

    mu = 1.0
    sigma2_ = 0.05
    mus_full = np.array([
        [- mu, 0.0],
        [- mu, mu],
        [- mu, -mu],
        [- mu, 2 * mu],
        [- mu, - 2 * mu],
        [mu, 0.0],
        [mu, mu],
        [mu, -mu],
        [mu, 2 * mu],
        [mu, - 2 * mu],
    ])

    dist_to_means = euclidean_distance_einsum(x, mus_full)
    out = logsumexp(-dist_to_means / (2 * sigma2_), axis=1)

    return out

  def plot_distribution(ax, xrng=None, yrng=None, cmap=None):
    xrng = [-2, 2] if xrng is None else xrng
    yrng = [-2, 2] if yrng is None else yrng

    rngx = np.linspace(xrng[0], xrng[1], 100)
    rngy = np.linspace(yrng[0], yrng[1], 100)
    xx, yy = np.meshgrid(rngx, rngy)
    coords = np.dstack([xx, yy])
    coords = coords.reshape(-1, 2)
    log_ps = log_p_pure(coords)
    z = log_ps.reshape(100, 100)
    z = np.exp(z)
    ax.contourf(xx, yy, z, levels=50, cmap=cmap)

  return log_p_pure, None, plot_distribution


def toy_gmm(n_comp=8, std=0.075, radius=0.5):
  """Ring of 2D Gaussians. Returns energy and sample functions."""

  means_x = np.cos(2 * np.pi *
                   np.linspace(0, (n_comp - 1) / n_comp, n_comp)).reshape(
                       n_comp, 1, 1, 1)
  means_y = np.sin(2 * np.pi *
                   np.linspace(0, (n_comp - 1) / n_comp, n_comp)).reshape(
                       n_comp, 1, 1, 1)
  mean = radius * np.concatenate((means_x, means_y), axis=1)
  weights = np.ones(n_comp) / n_comp

  def neg_energy(x):
    means = jnp.array(mean.reshape((-1, 1, 2)))
    c = np.log(n_comp * 2 * np.pi * std**2)
    f = -jax.nn.logsumexp(
        jnp.sum(-0.5 * jnp.square((x - means) / std), axis=2), axis=0) + c
    return -f

  def sample(n_samples):
    toy_sample = np.zeros(0).reshape((0, 2, 1, 1))
    sample_group_sz = np.random.multinomial(n_samples, weights)
    for i in range(n_comp):
      sample_group = mean[i] + std * np.random.randn(
          2 * sample_group_sz[i]).reshape(-1, 2, 1, 1)
      toy_sample = np.concatenate((toy_sample, sample_group), axis=0)
      np.random.shuffle(toy_sample)
    return toy_sample[:, :, 0, 0]

  return neg_energy, sample


def toy_rings(n_comp=4, std=0.075, radius=0.5):
  """Mixture of rings distribution. Returns energy and sample functions."""

  weights = np.ones(n_comp) / n_comp

  def neg_energy(x):
    r = jnp.sqrt((x[:, 1] ** 2) + (x[:, 0] ** 2))[:, None]
    means = (jnp.arange(1, n_comp + 1) * radius)[None, :]
    c = jnp.log(n_comp * 2 * np.pi * std**2)
    f = -jax.nn.logsumexp(-0.5 * jnp.square((r - means) / std), axis=1) + c
    return -f

  def sample(n_samples):
    toy_sample = np.zeros(0).reshape((0, 2, 1, 1))
    sample_group_sz = np.random.multinomial(n_samples, weights)
    for i in range(n_comp):
      sample_radii = radius*(i+1) + std * np.random.randn(sample_group_sz[i])
      sample_thetas = 2 * np.pi * np.random.random(sample_group_sz[i])
      sample_x = sample_radii.reshape(-1, 1) * np.cos(sample_thetas).reshape(
          -1, 1)
      sample_y = sample_radii.reshape(-1, 1) * np.sin(sample_thetas).reshape(
          -1, 1)
      sample_group = np.concatenate((sample_x, sample_y), axis=1)
      toy_sample = np.concatenate(
          (toy_sample, sample_group.reshape((-1, 2, 1, 1))), axis=0)
    return toy_sample[:, :, 0, 0]

  return neg_energy, sample
