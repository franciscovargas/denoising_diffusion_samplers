"""Module containing sampler objects.
"""
from typing import Union

import haiku as hk

import jax
from jax import numpy as np

from dds.discretisation_schemes import uniform_step_scheme

from dds.solvers import sdeint_udp_ito_em_scan_ou
from dds.solvers import sdeint_udp_ito_em_scan_ou_logged


class AugmentedAbstractFollmerSDEUDP(hk.Module):
  """Basic pinned brownian motion prior STL based sampler. This implements PIS.
  """
  alpha: Union[float, np.ndarray]
  sigma: Union[float, np.ndarray]
  dim: int

  # drift_network #: Callable[[], hk.Module]

  def __init__(
      self, sigma, dim, drift_network, tfinal=1, dt=0.05, target=None,
      step_fac=100, step_scheme=uniform_step_scheme,
      alpha=1, detach_dif_path=False, detach_stl_drift=False,
      detach_dritf_path=False, tpu=True, diff_net=None,
      name="BM_STL_follmer_sampler"
  ):
    super().__init__(name=name)
    self.gamma = (sigma)**2

    self.dtype = np.float32 if tpu else np.float64

    self.dim = dim
    self.aug_dim = dim * 2
    self.drift_network = drift_network()
    self.step_scheme = step_scheme

    self.tfinal = tfinal
    self.dt = dt

    self.target = target  # Target distribution used by networks

    # flags which can be useful for different estimators (e.g. CE, Vargrad etc)
    self.detach_drift_path = detach_dritf_path  # Useful for CE/Vargrad
    self.detach_dif_path = detach_dif_path      # Useful for CE/Vargrad
    self.detach_drift_stoch = detach_stl_drift  # For STL estimator

    # Detached network for STL estimator
    self.detached_drift = self.drift_network.__class__(
        architecture_specs=self.drift_network.architecture_specs,
        dim=self.drift_network.state_dim,
        name="stl_detach"
    )

    # For annealing purposes in the detached network
    self.detached_drift.ts = self.step_scheme(
        0, self.tfinal, self.dt, dtype=self.dtype,
        **dict())

  def __call__(self, batch_size, is_training=True, dt=None, ode=False):
    key = hk.next_rng_key()
    dt = self.dt if dt is None or is_training else dt
    return self.sample_aug_trajectory(
        batch_size, key, dt=dt, ode=ode)

  def sample_aug_trajectory(self, batch_size, key, dt, ode):
    pass

  def init_sample(self, n, key):
    r"""Initialises Y_0 for the SDE to \\delta_0 (Pinned Brownain Motion).

    Args:
      n: number of samples (e.g. number of samples to estimate elbo)
      key: random key.

    Returns:
      initialisation array.
    """
    pass


class AugmentedOUDFollmerSDEUDP(AugmentedAbstractFollmerSDEUDP):
  """Basic stationary underdamped OU prior based sampler (stl augmented).
  """
  alpha: Union[float, np.ndarray]
  sigma: Union[float, np.ndarray]
  dim: int

  def __init__(
      self, sigma, dim, drift_network, tfinal=1, dt=0.05, target=None,
      step_fac=100, step_scheme=uniform_step_scheme,
      alpha=1, m=1, detach_dif_path=False, tpu=True, detach_stl_drift=False,
      detach_dritf_path=False, log=False,
      diff_net=None, name="Eact_OU_STL_follmer_sampler"
  ):
    super().__init__(
        sigma, dim, drift_network,
        step_scheme=step_scheme,
        target=target, detach_dritf_path=detach_dritf_path,
        detach_stl_drift=detach_stl_drift, tpu=tpu,
        detach_dif_path=detach_dif_path, tfinal=tfinal, dt=dt,
        diff_net=diff_net, name=name)
    self.log = log
    self.alpha = np.exp(alpha) if self.log else alpha
    self.dim = dim
    self.sigma = sigma

    self.m = m

  def init_sample(self, n, key):
    y_0 = jax.random.normal(key, (n, self.dim)) * self.sigma
    q_0 = jax.random.normal(key, (n, self.dim)) * np.sqrt(self.m)
    return np.concatenate((y_0, q_0), axis=-1)

  def f_aug(self, y, t, args):
    """See base class."""
    t_ = t * np.ones((y.shape[0], 1))

    y_no_aug = y[..., :self.dim * 2]
    # import pdb; pdb.set_trace()

    u_t = self.drift_network(y_no_aug, t_, self.target)  # [..., :self.dim]

    u_t_normsq = ((u_t)**2).sum(axis=-1)[..., None]

    n, _ = y_no_aug.shape
    zeros = np.zeros((n, 1))

    state = np.concatenate((u_t, zeros, u_t_normsq), axis=-1)
    return state

  def g_aug(self, y, t, args):
    """See base class."""
    t_ = t * np.ones((y.shape[0], 1))
    y_no_aug_plus_lat = y[..., :self.dim * 2]
    y_no_aug = y[..., self.dim: self.dim * 2]

    n, _ = y_no_aug.shape

    gamma_t = self.sigma * np.ones_like(y_no_aug)

    zeros = np.zeros((n, 1))

    if self.detach_drift_stoch:
      u_t = self.detached_drift(y_no_aug_plus_lat, t_, self.target)
    else:
      u_t = self.drift_network(y_no_aug_plus_lat, t_, self.target)

    out = np.concatenate((gamma_t, u_t, zeros), axis=-1)

    return out

  def sample_aug_trajectory(
      self, batch_size, key, dt=0.05, ode=False, **_):
    y0 = self.init_sample(batch_size, key)

    zeros = np.zeros((batch_size, 1))
    y0_aug = np.concatenate((y0, zeros, zeros), axis=1)

    # notice no g_prod as that is handled internally by this specialised
    # ou based sampler.
    integrator = sdeint_udp_ito_em_scan_ou_logged if self.log else sdeint_udp_ito_em_scan_ou
    param_trajectory, ts = integrator(
        self.dim, self.alpha, self.f_aug, self.g_aug, y0_aug, key, dt=dt,
        end=self.tfinal, step_scheme=self.step_scheme, m=self.m
    )

    latent_traj = param_trajectory[:, :, :self.dim]
    girsanov_terms = param_trajectory[:, :, -2:]

    out_param_trajectory = np.concatenate((latent_traj, girsanov_terms),
                                          axis=-1)
    return out_param_trajectory, ts

