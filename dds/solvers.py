"""Numerical SDE Solvers library.
"""
import haiku as hk
import jax

from jax import numpy as np
# from jax.experimental import ode

from dds.discretisation_schemes import uniform_step_scheme

from dds.hutchinsons import get_div_fn


def sdeint_ito_em_scan(
    f, g, y0, rng, args=(), dt=1e-06, g_prod=None,
    step_scheme=uniform_step_scheme, start=0, end=1, dtype=np.float32,
    scheme_args=None):
  """Vectorised (scan based) implementation of EM discretisation.

  Args:
    f: drift coeficient - vector field
    g: diffusion coeficient
    y0: samples from initial dist
    rng: rng key
    args: optional arguments for f and g
    dt: discertisation step
    g_prod: multiplication routine for diff coef / noise, defaults to hadamard
    step_scheme: how to spread out the integration steps defaults to uniform
    start: start time defaults to 0
    end: end time defaults to 1
    dtype: float32 or 64 (for tpu)
    scheme_args: args for step scheme

  Returns:
    Trajectory augmented with the path objective (Time x batch_size x (dim + 1))
  """

  scheme_args = scheme_args if scheme_args is not None else {}
  if g_prod is None:

    def g_prod(y, t, args, noise):
      out = g(y, t, args) * noise
      return out

  ts = step_scheme(start, end, dt, dtype=dtype, **scheme_args)

  y_pas = y0
  t_pas = ts[0]

  def euler_step(ytpas, t_):
    (y_pas, t_pas, rng) = ytpas

    delta_t = t_ - t_pas

    this_rng, rng = jax.random.split(rng)
    noise = jax.random.normal(this_rng, y_pas.shape, dtype=dtype)

    y = y_pas + f(y_pas, t_pas, args) * delta_t + g_prod(
        y_pas, t_pas, args, noise
    ) * np.sqrt(delta_t)

    # t_pas = t_
    # y_pas = y
    out = (y, t_, rng)
    return out, y

  _, ys = hk.scan(euler_step, (y_pas, t_pas, rng), ts[1:])

  return np.swapaxes(np.concatenate((y0[None], ys), axis=0), 0, 1), ts


def sdeint_ito_em_scan_ou(
    dim, alpha, f, g, y0, rng, args=(), dt=1e-06,
    step_scheme=uniform_step_scheme, start=0, end=1, dtype=np.float32,
    scheme_args=None, ddpm_param=True):
  """Vectorised (scan based) implementation Exact OU based discretisation.

  Using this objective allows (and computes) a simplified Girsanov like ELBO
  for the OU at equilibrium based reference process. In short this is the
  discretisation for the denoising diffusion based sampler.

  Args:
    dim: dimension of the input statespace (non augmented)
    alpha: ou drift coef
    f: drift coeficient - vector field
    g: diffusion coeficient
    y0: samples from initial dist
    rng: rng key
    args: optional arguments for f and g
    dt: discertisation step
    step_scheme: how to spread out the integration steps defaults to uniform
    start: start time defaults to 0
    end: end time defaults to 1
    dtype: float32 or 64 (for tpu)
    scheme_args: args for step scheme
    ddpm_param: type of parametrisation for exp(-alpha dt )

  Returns:
    Trajectory augmented with the path objective (Time x batch_size x (dim + 1))
  """

  scheme_args = scheme_args if scheme_args is not None else {}
  ts = step_scheme(start, end, dt, dtype=dtype, **scheme_args)
  detach = True if args and "detach" in args else False

  y_pas = y0
  t_pas = ts[0]

  def euler_step(ytpas, t_):
    (y_pas, t_pas, rng) = ytpas

    delta_t = t_ - t_pas

    if ddpm_param:  # ddpmp styled parametrisation
      # could have done alpha_k = np.exp(-alpha * delta_t) and
      # beta_k = np.sqrt(1.0 - alpha_k**2) but the scale is harder to compare to
      # PIS and grid search so we are sticking to this notation/setup which is
      # more inline with DDPM and SDE score matching
      beta_k = np.clip(alpha * np.sqrt(delta_t), 0, 1)
      alpha_k = np.sqrt(1.0 - beta_k**2)
    else:        # more formal OU closed form transition looking parametrisation
      alpha_k = np.clip(np.exp(-alpha * delta_t), 0, 0.99999)
      beta_k = np.sqrt(1.0 - alpha_k**2)

    this_rng, rng = jax.random.split(rng)
    noise = jax.random.normal(this_rng, y_pas.shape, dtype=dtype)

    y_pas_naug = y_pas[:, :dim]
    g_aug = g(y_pas, t_pas, args)
    f_aug = f(y_pas, t_pas, args)

    # State update
    y_naug = y_pas_naug * alpha_k + f_aug[:, :dim] * beta_k**2 + (
        g_aug[:, :dim] * noise[:, :dim]) * beta_k

    # Stoch int (detached STL term) update
    u_dw = np.squeeze(y_pas[:, dim:-1]) + np.einsum(
        "ij,ij->i", g_aug[:, dim:-1], noise[:, :dim]) * beta_k

    # Girsanov (quadratic) term update
    u_sq = y_pas[:, -1] + f_aug[:, -1] * beta_k**2
    
    # For cross entropy refinement
    if detach:
      f_aug_att = f(y_pas, t_pas, [])
      u_sq_att = y_pas[:, -1] + f_aug_att[:, -1] * beta_k**2
    else:
      u_sq_att = u_sq

    y = np.concatenate((y_naug,
                        u_dw[..., None],
                        u_sq_att[..., None],
                        u_sq[..., None]), axis=-1)

    # t_pas = t_
    # y_pas = y
    out = (y, t_, rng)
    return out, y

  _, ys = hk.scan(euler_step, (y_pas, t_pas, rng), ts[1:])

  return np.swapaxes(np.concatenate((y0[None], ys), axis=0), 0, 1), ts


def odeint_em_scan_ou(
    dim, alpha, f, g, y0, rng, args=(), dt=1e-06,
    step_scheme=uniform_step_scheme, start=0, end=1, dtype=np.float32,
    scheme_args=None, ddpm_param=True):
  """Vectorised (scan based) implementation Exact OU based discretisation.

  Using this objective allows (and computes) a simplified Girsanov like ELBO
  for the OU at equilibrium based reference process. In short this is the
  discretisation for the denoising diffusion based sampler.

  Args:
    dim: dimension of the input statespace (non augmented)
    alpha: ou drift coef
    f: drift coeficient - vector field
    g: diffusion coeficient
    y0: samples from initial dist
    rng: rng key
    args: optional arguments for f and g
    dt: discertisation step
    step_scheme: how to spread out the integration steps defaults to uniform
    start: start time defaults to 0
    end: end time defaults to 1
    dtype: float32 or 64 (for tpu)
    scheme_args: args for step scheme
    ddpm_param: type of parametrisation for exp(-alpha dt )

  Returns:
    Trajectory augmented with the path objective (Time x batch_size x (dim + 1))
  """

  scheme_args = scheme_args if scheme_args is not None else {}
  ts = step_scheme(start, end, dt, dtype=dtype, **scheme_args)

  y_pas = y0
  t_pas = ts[0]
  _, _ = g, rng

  f_div = get_div_fn(f, rng, y0[:, :dim].shape)

  def euler_step(ytpas, t_):
    (y_pas, t_pas, k, rng) = ytpas

    delta_t = t_ - t_pas
    # delta_t2 = ts[k+1] - t_

    if ddpm_param:
      beta_k_sq = np.clip(alpha**2 * delta_t, 0, 1)
      alpha_k = np.sqrt(1.0 - beta_k_sq)
    else:
      alpha_k = np.clip(np.exp(-alpha * delta_t), 0, 0.99999)
      beta_k_sq = (1.0 - alpha_k**2)

    # this_rng, rng = jax.random.split(rng)
    # noise = jax.random.normal(this_rng, y_pas.shape, dtype=dtype)

    y_pas_naug = y_pas[:, :dim]
    # g_aug = g(y_pas, t_pas, args)
    f_aug = f(y_pas, t_pas, args)

    # State update (deterministic ODE)
    # beta_^2 \approx 2 delta beta_cont_t

    # Heuns's method
    a1, a2 = (0.5, 0.5)
    p, q = (1.0, 1.0)
    k1 = f_aug[:, :]
    y_pass_prime = y_pas + beta_k_sq * 0.5  * k1 * q
    t_n = t_pas + p * delta_t
    k2 = f(y_pass_prime, t_n, args)
    y_naug = y_pas_naug + beta_k_sq * 0.5 * (
        a2 * k2[:, :dim] + a1 * k1[:, :dim])

    # Hutchinsons trace estimator (Again integrating with Heun)
    a1_htch, a2_htch = (0.5, 0.5)
    k1_tr = f_div(y_pas, t_pas)
    k2_tr = f_div(y_pass_prime, t_n)
    # u_sq = y_pas[:, -1] + delta_t * alpha * k1_tr
    u_sq = y_pas[:, -1] +  beta_k_sq * 0.5 * (a1_htch * k1_tr + a2_htch * k2_tr)

    y = np.concatenate((y_naug, np.zeros((y0.shape[0], 1)), u_sq[..., None]),
                       axis=-1)

    # t_pas = t_
    # y_pas = y
    k += 1
    out = (y, t_, k, rng)
    return out, y

  # def ode_func(y, t):
  #   f_aug = f(y, t, args)
  #   f_noaug = alpha * f_aug[:, :]
  #   return f_noaug

  k = 0
  # if ddpm_param:
  _, ys = hk.scan(euler_step, (y_pas, t_pas, k, rng), ts[1:])

  # Reverse ODE approach, kinda works not as good.
  # ts_back = np.flip(ts, axis=0)
  # t_final = ts_back[0]
  # y_tfinal = ys[-1, :, :]
  # _, ys_back = hk.scan(euler_step, (y_tfinal, t_final, k, rng), ts_back[1:])

  # trace_back = -ys_back[:, :, -1][..., None]
  # trace_front = ys[:, :, -1][..., None]
  # ys_state = ys[:, :, :dim]
  # y0prime = ys_back[-1, :, :]

  # ys_out = np.concatenate((ys_state, trace_back, trace_front), axis=-1)

  return np.swapaxes(np.concatenate((y0[None], ys), axis=0), 0, 1), ts


def sdeint_udp_ito_em_scan_ou(
    dim, alpha, f, g, y0, rng, args=(), dt=1e-06, m=1,
    step_scheme=uniform_step_scheme, start=0, end=1, dtype=np.float32,
    scheme_args=None, _=False):
  """Vectorised (scan based) implementation Exact OU based discretisation.

  Using this objective allows (and computes) a simplified Girsanov like ELBO
  for the OU at equilibrium based reference process. In short this is the
  discretisation for the denoising diffusion based sampler.

  Args:
    dim: dimension of the input statespace (non augmented)
    alpha: ou drift coef
    f: drift coeficient - vector field
    g: diffusion coeficient
    y0: samples from initial dist
    rng: rng key
    args: optional arguments for f and g
    dt: discertisation step
    m: mass for Hamiltonian system, the bane of my life.
    step_scheme: how to spread out the integration steps defaults to uniform
    start: start time defaults to 0
    end: end time defaults to 1
    dtype: float32 or 64 (for tpu)
    scheme_args: args for step scheme

  Returns:
    Trajectory augmented with the path objective (Time x batch_size x (dim + 1))
  """

  scheme_args = scheme_args if scheme_args is not None else {}
  ts = step_scheme(start, end, dt, dtype=dtype, **scheme_args)

  y_pas = y0
  t_pas = ts[0]

  def euler_step(ytpas, t_):
    (y_pas, t_pas, rng) = ytpas

    delta_t = t_ - t_pas

    # we need the exp param
    # alpha_k = np.clip(np.exp(-alpha * delta_t), 0, 0.99999)
    # beta_k = np.sqrt(1.0 - alpha_k**2)

    alpha_k = 1 - np.clip(np.exp(-2 * alpha * delta_t), 0, 0.99999)

    rt_1_mnsalpha_k = np.sqrt(1.0 - alpha_k)

    this_rng, rng = jax.random.split(rng)
    noise = jax.random.normal(this_rng, y_pas.shape, dtype=dtype)

    y_latent = y_pas[:, :dim]
    q_obs = y_pas[:, dim: 2 * dim]
    udw_old = y_pas[:, 2 * dim: -1]

    # Only works for constant (homogenous in time and space) sigma atm
    g_aug = g(y_pas, t_pas, args)
    sigma = g_aug[:, :dim]

    # Exact Phi_flip o Phi o Phi_flip  update
    # sm =
    theta = delta_t / (sigma * np.sqrt(m))
    y_latent_next = y_latent * np.cos(theta) - q_obs * np.sin(theta) * (
        sigma / np.sqrt(m))
    q_prime = y_latent * np.sin(theta) * (np.sqrt(m) /
                                          sigma) + q_obs * np.cos(theta)

    y_pas_prime = np.concatenate((y_latent_next, q_prime), axis=-1)
    f_aug = f(y_pas_prime, t_pas, args)

    f_ = f_aug[:, :dim]

    # Observable state update
    lambda_k = (1.0 - rt_1_mnsalpha_k)
    q_new = rt_1_mnsalpha_k * (q_prime +
                               2.0 * m * lambda_k * np.sqrt(alpha_k) * f_) + (
                                   noise[:, :dim] * np.sqrt(m * alpha_k))

    # Stoch int (detached STL term) update
    udw_new = np.squeeze(udw_old) + np.einsum(
        "ij,ij->i", f_,
        noise[:, :dim]) * np.sqrt(m) * rt_1_mnsalpha_k * lambda_k * 2.0

    # Girsanov (quadratic) term update
    u_sq = y_pas[:, -1] + (
        f_aug[:, -1] * rt_1_mnsalpha_k**2 * lambda_k**2 * m * 2.0)

    y = np.concatenate(
        (y_latent_next, q_new, udw_new[..., None], u_sq[..., None]), axis=-1)

    out = (y, t_, rng)
    return out, y

  _, ys = hk.scan(euler_step, (y_pas, t_pas, rng), ts[1:])

  return np.swapaxes(np.concatenate((y0[None], ys), axis=0), 0, 1), ts


def sdeint_udp_ito_em_scan_ou_logged(
    dim, alpha, f, g, y0, rng, args=(), dt=1e-06, m=1,
    step_scheme=uniform_step_scheme, start=0, end=1, dtype=np.float32,
    scheme_args=None, _=False):
  """Vectorised (scan based) implementation Exact OU based discretisation.

  Using this objective allows (and computes) a simplified Girsanov like ELBO
  for the OU at equilibrium based reference process. In short this is the
  discretisation for the denoising diffusion based sampler.

  Args:
    dim: dimension of the input statespace (non augmented)
    alpha: ou drift coef
    f: drift coeficient - vector field
    g: diffusion coeficient
    y0: samples from initial dist
    rng: rng key
    args: optional arguments for f and g
    dt: discertisation step
    m: mass for Hamiltonian system, the bane of my life.
    step_scheme: how to spread out the integration steps defaults to uniform
    start: start time defaults to 0
    end: end time defaults to 1
    dtype: float32 or 64 (for tpu)
    scheme_args: args for step scheme

  Returns:
    Trajectory augmented with the path objective (Time x batch_size x (dim + 1))
  """

  scheme_args = scheme_args if scheme_args is not None else {}
  ts_scale = step_scheme(start, end, dt, dtype=dtype, **scheme_args)
  ts = uniform_step_scheme(start, end, dt, dtype=dtype, **scheme_args)

  y_pas = y0
  t_pas = ts[0]
  k_pas = 0

  def euler_step(ytpas, t_):
    (y_pas, t_pas, k, rng) = ytpas

    delta_t = t_ - t_pas
    scale = -np.log(alpha) - np.log(ts_scale[k+1]  - ts_scale[k])

    alpha_k = 1 - np.clip(np.exp(-2 * scale * delta_t), 0, 0.99999)

    rt_1_mnsalpha_k = np.sqrt(1.0 - alpha_k)

    this_rng, rng = jax.random.split(rng)
    noise = jax.random.normal(this_rng, y_pas.shape, dtype=dtype)

    y_latent = y_pas[:, :dim]
    q_obs = y_pas[:, dim: 2 * dim]
    udw_old = y_pas[:, 2 * dim: -1]

    # Only works for constant (homogenous in time and space) sigma atm
    g_aug = g(y_pas, t_pas, args)
    sigma = g_aug[:, :dim]

    # Exact Phi_flip o Phi o Phi_flip  update
    theta = delta_t / (sigma * np.sqrt(m))
    y_latent_next = y_latent * np.cos(theta) - q_obs * np.sin(theta) * (
        sigma / np.sqrt(m))
    q_prime = y_latent * np.sin(theta) * (np.sqrt(m) /
                                          sigma) + q_obs * np.cos(theta)

    y_pas_prime = np.concatenate((y_latent_next, q_prime), axis=-1)
    f_aug = f(y_pas_prime, t_pas, args)

    f_ = f_aug[:, :dim]

    # Old Leapfrog update
    # q_tilde = q_obs + 0.5 * delta_t * y_latent / sigma**2
    # y_latent_next = y_latent  - delta_t * q_tilde
    # q_prime = q_tilde + 0.5 * delta_t * y_latent_next / sigma**2

    # Observable state update
    lambda_k = (1.0 - rt_1_mnsalpha_k)
    q_new = rt_1_mnsalpha_k * (q_prime +
                               2.0 * m * lambda_k * np.sqrt(alpha_k) * f_) + (
                                   noise[:, :dim] * np.sqrt(m * alpha_k))

    # Stoch int (detached STL term) update
    udw_new = np.squeeze(udw_old) + np.einsum(
        "ij,ij->i", f_,
        noise[:, :dim]) * np.sqrt(m) * rt_1_mnsalpha_k * lambda_k * 2.0

    # Girsanov (quadratic) term update
    u_sq = y_pas[:, -1] + (
        f_aug[:, -1] * rt_1_mnsalpha_k**2 * lambda_k**2 * m * 2.0)

    y = np.concatenate(
        (y_latent_next, q_new, udw_new[..., None], u_sq[..., None]), axis=-1)

    out = (y, t_, k + 1, rng)
    return out, y

  _, ys = hk.scan(euler_step, (y_pas, t_pas, k_pas, rng), ts[1:])

  return np.swapaxes(np.concatenate((y0[None], ys), axis=0), 0, 1), ts
