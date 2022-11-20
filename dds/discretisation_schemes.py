"""Time discretisation schemes.
"""
from jax import numpy as np


def power_fn_step_scheme(start, end, dt, exp=0.9, dtype=np.float32, **_):
  """Exponent decay step scheme from Zhang et al. 2021.

  Args:
    start: start time defaults to 0
    end: end time defaults to 1
    dt: number of steps to divide grid into
    exp: exponent
    dtype: for tpu support
    **_: placeholder to handle different scheme args

  Returns:
    time grid
  """
  n_steps = int((end- start) / dt)
  base = np.linspace(start ** exp, end ** exp, n_steps, dtype=dtype)
  return np.power(base, 1.0 / exp)


def exp_fn_step_scheme(
    start, end, dt, base=2.71828, dtype=np.float32, **_):
  """Exponential decay step scheme from Zhang et al. 2021.

  Args:
    start: start time defaults to 0
    end: end time defaults to 1
    dt: number of steps to divide grid into
    base: base
    dtype: for tpu support
    **_: placeholder to handle different scheme args

  Returns:
    time grid
  """
  n_steps = int((end - start) / dt)
  dts = base**(-5 * 2 * np.linspace(start, end, n_steps, dtype=dtype) / end)

  dts /= dts.sum()
  dts *= end

  dts_out = np.concatenate((np.array([start]), np.cumsum(dts)))
  return dts_out


def cos_sq_fn_step_scheme(
    start, end, dt, s=0.008, dtype=np.float32, **_):
  """Exponential decay step scheme from Nichol and Dhariwal 2021.

  Args:
    start: start time defaults to 0
    end: end time defaults to 1
    dt: number of steps to divide grid into
    s: shift to ensure non 0
    dtype: for tpu support
    **_: placeholder to handle different scheme args

  Returns:
    time grid
  """
  n_steps = int((end - start) / dt)

  pre_phase = np.linspace(start, end, n_steps, dtype=dtype) / end
  phase = ((pre_phase + s) / (1 + s)) * np.pi * 0.5
  # Note this multiples small numbers however implemented it more stably
  # (removed sqrt from solver only sqrd here) and it made no difference to
  # results
  dts = np.cos(phase)**4

  dts /= dts.sum()
  dts *= end  # We normalise s.t. \sum_k \beta_k = T (where beta_k = b_m*cos^4)

  dts_out = np.concatenate((np.array([start]), np.cumsum(dts)))
  return dts_out


def triangle_step_scheme(
    start, end, dt, dt_max=0.2, dt_min=0.01, f=np.exp, dtype=np.float32, **_):
  """Triangle step scheme from Bortoli et al. 2021.

  Args:
    start: start time defaults to 0
    end: end time defaults to 1
    dt: number of steps to divide grid into
    dt_max: largest dt
    dt_min: smallest dt
    f: non lineartiy to apply to increments
    dtype: for tpu support
    **_: placeholder to handle different scheme args

  Returns:
    time grid
  """

  n_steps = int((end- start) / dt)
  ts = np.linspace(start, end, n_steps, dtype=dtype)

  m = 2 * (dt_max - dt_min) / (n_steps)  # gradient

  uts = f((dt_max - m * np.abs(ts[1:] - (start + 0.5 * n_steps * dt))))
  uts = start + (uts * end) / uts.sum()
  dts = np.concatenate((ts[0][None], np.cumsum(uts)))
  # import pdb; pdb.set_trace()
  return dts


def linear_step_scheme(
    start, end, dt, dt_min=0.0001, dtype=np.float32, **_):
  """Linear step scheme from Ho et al. 2020.

  Args:
    start: start time defaults to 0
    end: end time defaults to 1
    dt: number of steps to divide grid into and dt_max
    dt_min: smallest dt
    dtype: for tpu support
    **_: placeholder to handle different scheme args

  Returns:
    time grid
  """
  dt_max = dt
  n_steps = int((end- start) / dt)
  ts = np.linspace(start, end, n_steps, dtype=dtype)

  dt_ = np.abs(ts[1]- ts[0])
  m = (dt_max - dt_min) / (end - dt_)  # gradient

  uts = dt_max - m * np.abs(ts[:-1])
  uts = start + (uts * end) / uts.sum()
  dts = np.concatenate((ts[0][None], np.cumsum(uts)))
  return dts


def linear_step_scheme_dds(start,
                           end,
                           dt,
                           dt_max=0.04,
                           dt_min=0.02,
                           dtype=np.float32,
                           **_):
  """Linear step scheme for Ho et al 2020 applied to sampling.

  2021.

  Args:
    start: start time defaults to 0
    end: end time defaults to 1
    dt: number of steps to divide grid into
    dt_max: largest dt
    dt_min: smallest dt
    dtype: for tpu support
    **_: placeholder to handle different scheme args

  Returns:
    time grid
  """

  n_steps = int((end- start) / dt)
  uts = np.linspace(dt_min, dt_max, n_steps, dtype=dtype)[::-1]

  dts = np.cumsum(uts)
  # import pdb; pdb.set_trace()
  return dts


def uniform_step_scheme_dds(start,
                            end,
                            dt,
                            dt_max=0.02,
                            dtype=np.float32,
                            **_):
  """Linear step scheme for Ho et al 2020 applied to sampling.

  2021.

  Args:
    start: start time defaults to 0
    end: end time defaults to 1
    dt: number of steps to divide grid into
    dt_max: largest dt
    dtype: for tpu support
    **_: placeholder to handle different scheme args

  Returns:
    time grid
  """

  n_steps = int((end- start) / dt)
  uts = np.ones((n_steps), dtype=dtype) * dt_max

  dts = np.cumsum(uts)  # np.concatenate((np.array([start]), np.cumsum(uts)))
  # import pdb; pdb.set_trace()
  return dts


def small_lst_step_scheme(start, end, dt, step_fac=100, dtype=np.float32, **_):
  """Scales the final step by a provided factor.

  Args:
    start: start time defaults to 0
    end: end time defaults to 1
    dt: 1/number of steps to divide grid into
    step_fac: final step factor
    dtype: for tpu support
    **_: placeholder to handle different scheme args

  Returns:
    time grid
  """
  n_steps = int((end- start) / dt)
  dt_last = dt / step_fac
  dt_pen = dt * (step_fac - 1)  / step_fac

  ts = np.linspace(start, end - dt, n_steps -1, dtype=dtype)

  t_pen = ts[-1] + dt_pen
  t_fin = t_pen + dt_last

  ts = np.concatenate((ts, t_pen[None]), axis=0)
  ts = np.concatenate((ts, t_fin[None]), axis=0)
  return ts


def uniform_step_scheme(start, end, dt, dtype=np.float32, **_):
  """Standard uniform scaling.

  Args:
    start: start time defaults to 0
    end: end time defaults to 1
    dt: 1/number of steps to divide grid into
    dtype: for tpu support
    **_: placeholder to handle different scheme args

  Returns:
    time grid
  """
  n_steps = int((end- start) / dt)

  ts = np.linspace(start, end, n_steps, dtype=dtype)

  return ts
