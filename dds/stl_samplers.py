"""Module containing sampler objects.
"""
from typing import Union

import haiku as hk

import jax
from jax import numpy as np

from dds.discretisation_schemes import uniform_step_scheme
from dds.drift_nets import OUDrift

from dds.solvers import odeint_em_scan_ou
from dds.solvers import sdeint_ito_em_scan
from dds.solvers import sdeint_ito_em_scan_ou


class AugmentedBrownianFollmerSDESTL(hk.Module):
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
      name="BM_STL_follmer_sampler", **_
  ):
    super().__init__(name=name)
    self.gamma = (sigma)**2

    self.dtype = np.float32 if tpu else np.float64

    self.dim = dim
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

  def __call__(
      self, batch_size, is_training=True,
      dt=None, ode=False):
    key = hk.next_rng_key()
    dt = self.dt if dt is None or is_training else dt
    return self.sample_aug_trajectory(
        batch_size, key, dt=dt, is_training=is_training, ode=ode)

  def init_sample(self, n, key):
    r"""Initialises Y_0 for the SDE to \\delta_0 (Pinned Brownain Motion).

    Args:
      n: number of samples (e.g. number of samples to estimate elbo)
      key: random key.

    Returns:
      initialisation array.
    """
    return np.zeros((n, self.dim))

  def f_aug(self, y, t, args):
    """Computes the drift of the SDE + augmented state space for loss.

    Computes drift and auxiliary variables for the SDE:
              dY_t = drift(Y_t,t)dt + gamma(t) dW_t

    Args:
      y: state spaces at times t.
      t: times corresponding to each y.
      args: unused (atm) empty placeholder (useful for some solvers).

    Returns:
      Augmented statespace of the form [drift(y,t), 0, ||drift(y,t)||^2/(2*C^2)]
    """
    t_ = t * np.ones((y.shape[0], 1))

    y_no_aug = y[..., :self.dim]

    u_t = self.drift_network(y_no_aug, t_, self.target)

    gamma_t = self.g_aug(y, t, args)[..., :self.dim]**2

    u_t_normsq = ((u_t)**2 / gamma_t).sum(axis=-1)[..., None] / 2.0

    n, _ = y_no_aug.shape
    zeros = np.zeros((n, 1))

    return np.concatenate((u_t, zeros, u_t_normsq), axis=-1)

  def g_aug(self, y, t, args):
    r"""Computes the diff coefficient of the SDE+augmented state space for loss.

    Computes drift and auxiliary variables for the SDE:
              dY_t = drift(Y_t,t)dt + gamma_t dW_t

    Args:
      y: state spaces at times t.
      t: times corresponding to each y.
      args: unused (atm) empty placeholder (useful for some solvers).

    Returns:
      Augmented statespace of the form. [\gamma(t), drift(y,t)/C, 0]
    """

    t_ = t * np.ones((y.shape[0], 1))
    y_no_aug = y[..., :self.dim]

    n, _ = y_no_aug.shape

    # gamma(t) = gamma (homegenous diff for first d dimensions).
    gamma_ = np.sqrt(self.gamma) * np.ones_like(y_no_aug)

    zeros = np.zeros((n, 1))  # for last dimmension (which is noiseless)

    # stl vs no stl, here we compute drift(y,t) for the dim:2*dim-1 locs of
    # the augmented state space (drift(y,t)/C).
    if self.detach_drift_stoch:
      u_t = self.detached_drift(y_no_aug, t_, self.target)
    else:
      u_t = self.drift_network(y_no_aug, t_, self.target)

    out = np.concatenate((gamma_, u_t / gamma_, zeros), axis=-1)

    return out

  def sample_aug_trajectory(self, batch_size, key, dt=0.05, rng=None, **_):
    y0 = self.init_sample(batch_size, key)

    zeros = np.zeros((batch_size, 1))
    y0_aug = np.concatenate((y0, zeros), axis=1)

    def g_prod(y, t, args, noise):
      """Defines how to compute the product between the aug diff coef and noise.

      This function specifies how the brownian noise is multiplied with the
      augmented noise state space defined by g_aug. This is used by the approac-
      hes based on euler approx (sdeint_ito_em_scan), for ou solvers this
      behaviour was reimplemented inside the solver.

      Args:
        y: state spaces at times t.
        t: times corresponding to each y.
        args: unused (atm) empty placeholder (useful for some solvers).
        noise: brownian noise (to be passed in solver).

      Returns:
        Returns g_aug(Y_t, t) * dW_t
      """
      g_aug = self.g_aug(y, t, args)

      # We assume diagonal noise and thus g(t) dW_t is elementwise
      gdw = g_aug[:, :self.dim] * noise[:, :self.dim]

      # here we compute drift(Y_t, t)^T dW_t (needed for the IS refinement)
      udw = np.einsum("ij,ij->i",
                      g_aug[:, self.dim:-1],
                      noise[:, :self.dim])

      # the last coordinate is a 0 as it evolves ||u(Y_t,t)||^2/C noiselessly
      zeros = 0.0 * g_aug[:, -1] * noise[:, -1]

      return  np.concatenate((gdw, udw[..., None], zeros[..., None]), axis=-1)

    param_trajectory, ts = sdeint_ito_em_scan(
        self.f_aug, self.g_aug, y0_aug, key, dt=dt,
        g_prod=g_prod, end=self.tfinal, step_scheme=self.step_scheme,
        dtype=self.dtype
    )

    return param_trajectory, ts


class AugmentedOUFollmerSDESTL(AugmentedBrownianFollmerSDESTL):
  """Basic stationary OU prior based sampler (stl augmented).

  Uses Euler approximation (innacurate/bit unstable) for forward kerenl.
  """
  alpha: Union[float, np.ndarray]
  sigma: Union[float, np.ndarray]
  dim: int

  prior_drift: OUDrift
  # drift_network: Callable[[], hk.Module]

  def __init__(
      self, sigma, dim, drift_network, tfinal=1, dt=0.05, target=None,
      step_fac=100, step_scheme=uniform_step_scheme,
      alpha=1, detach_dif_path=False, tpu=True, detach_stl_drift=False,
      detach_dritf_path=False,
      diff_net=None, name="OU_STL_follmer_sampler", **_
  ):
    super().__init__(
        sigma, dim, drift_network,
        step_scheme=step_scheme,
        target=target, detach_dritf_path=detach_dritf_path,
        detach_stl_drift=detach_stl_drift, tpu=tpu,
        detach_dif_path=detach_dif_path, tfinal=tfinal, dt=dt,
        diff_net=diff_net, name=name)
    self.alpha = alpha
    self.sigma = sigma

    sqrt2alpha = np.sqrt(2 * alpha)
    self.gamma = (sqrt2alpha * self.sigma)**2

    self.prior_drift = OUDrift(
        alpha=alpha, sigma=sigma, dim=dim
    )

  def init_sample(self, n, key):
    """Initialises Y_0 for the SDE to the steady state dist N(0, sigma^2).

    Args:
      n: number of samples (e.g. number of samples to estimate elbo)
      key: random key.

    Returns:
      initialisation array.
    """
    return jax.random.normal(key, (n, self.dim)) * self.sigma

  def f_aug(self, y, t, args, detach=False):
    """See base class."""
    t_ = t * np.ones((y.shape[0], 1))

    y_no_aug = y[..., :self.dim]

    b_t = self.prior_drift(y_no_aug, t_)
   
    u_t = (self.detached_drift(
        y_no_aug, t_, self.target) 
           if detach else self.drift_network(y_no_aug, t_, self.target))

    gamma_t_sq = self.g_aug(y, t, args)[..., :self.dim]**2

    u_t_normsq = ((u_t - b_t)**2 / gamma_t_sq).sum(axis=-1)[..., None] / 2.0

    n, _ = y_no_aug.shape
    zeros = np.zeros((n, 1))

    state = np.concatenate((u_t, zeros, u_t_normsq), axis=-1)
    return state

  def g_aug(self, y, t, args):
    """See base class."""
    t_ = t * np.ones((y.shape[0], 1))
    y_no_aug = y[..., :self.dim]

    n, _ = y_no_aug.shape

    gamma_t = np.sqrt(self.gamma) * np.ones_like(y_no_aug)

    zeros = np.zeros((n, 1))

    b_t = self.prior_drift(y_no_aug, t_)

    if self.detach_drift_stoch:
      u_t = self.detached_drift(y_no_aug, t_, self.target)
    else:
      u_t = self.drift_network(y_no_aug, t_, self.target)

    delta_t = (u_t - b_t)  / gamma_t

    out = np.concatenate((gamma_t, delta_t, zeros), axis=-1)

    return out


class AugmentedOUDFollmerSDESTL(AugmentedBrownianFollmerSDESTL):
  """Basic stationary OU prior based sampler (stl augmented).
  """
  alpha: Union[float, np.ndarray]
  sigma: Union[float, np.ndarray]
  dim: int

  def __init__(
      self, sigma, dim, drift_network, tfinal=1, dt=0.05, target=None,
      step_fac=100, step_scheme=uniform_step_scheme,
      alpha=1, detach_dif_path=False, tpu=True, detach_stl_drift=False,
      detach_dritf_path=False,
      diff_net=None, exp_bool=False, name="Eact_OU_STL_follmer_sampler", **_
  ):
    super().__init__(
        sigma, dim, drift_network,
        step_scheme=step_scheme,
        target=target, detach_dritf_path=detach_dritf_path,
        detach_stl_drift=detach_stl_drift, tpu=tpu,
        detach_dif_path=detach_dif_path, tfinal=tfinal, dt=dt,
        diff_net=diff_net, name=name)
    self.alpha = alpha
    self.sigma = sigma
    self.exp_bool = exp_bool

  def init_sample(self, n, key):
    return jax.random.normal(key, (n, self.dim)) * self.sigma

  def f_aug(self, y, t, args):
    """See base class."""
    t_ = t * np.ones((y.shape[0], 1))

    y_no_aug = y[..., :self.dim]

    ode = True if args and "ode" in args else False
    detach = True if args and "detach" in args else False

#     u_t = self.drift_network(y_no_aug, t_, self.target, ode=ode)
    u_t = (self.detached_drift(
        y_no_aug, t_,
        self.target, ode=ode
    ) if detach else self.drift_network(
        y_no_aug, t_,
        self.target, ode=ode
    ))

    gamma_t_sq = self.g_aug(y, t, args)[..., :self.dim]**2

    u_t_normsq = ((u_t)**2 / gamma_t_sq).sum(axis=-1)[..., None] / 2.0

    n, _ = y_no_aug.shape
    zeros = np.zeros((n, 1))

    state = np.concatenate((u_t, zeros, u_t_normsq), axis=-1)
    return state

  def g_aug(self, y, t, args):
    """See base class."""
    t_ = t * np.ones((y.shape[0], 1))
    y_no_aug = y[..., :self.dim]

    n, _ = y_no_aug.shape

    gamma_t = self.sigma * np.ones_like(y_no_aug)

    zeros = np.zeros((n, 1))
    
    detach = True if args and "detach" in args else False
    if self.detach_drift_stoch or detach:
      u_t = self.detached_drift(y_no_aug, t_, self.target)
    else:
      u_t = self.drift_network(y_no_aug, t_, self.target)

    delta_t = (u_t)  / gamma_t

    out = np.concatenate((gamma_t, delta_t, zeros), axis=-1)

    return out

  def sample_aug_trajectory(
      self, batch_size, key, dt=0.05, rng=None, ode=False, **_):
    y0 = self.init_sample(batch_size, key)

    zeros = np.zeros((batch_size, 1))
    
    if odd:
        y0_aug = np.concatenate((y0, zeros, zeros), axis=1)
    else:
        y0_aug = np.concatenate((y0, zeros, zeros, zeros), axis=1)

    # notice no g_prod as that is handled internally by this specialised
    # ou based sampler.
    ddpm_param = not self.exp_bool
    integrator = odeint_em_scan_ou if ode else sdeint_ito_em_scan_ou
    param_trajectory, ts = integrator(
        self.dim, self.alpha, self.f_aug, self.g_aug, y0_aug, key, dt=dt,
        end=self.tfinal, step_scheme=self.step_scheme, ddpm_param=ddpm_param,
        dtype=self.dtype
    )

    return param_trajectory, ts
