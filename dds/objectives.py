"""Different stoch control objectives satisfying same HJB fixed points.
"""
import distrax
from jax import numpy as np
from jax import scipy as jscipy


def ou_terminal_loss(x_terminal, lnpi, sigma=1.0, tfinal=1.0, brown=False):
  """Terminal loss under OU reference prior at equilibrium.

  Can also be used for Brownian if you let sigma be the diff coef.

  Args:
      x_terminal: final time step samples from SDE
      lnpi: log target dist numerator
      sigma: stationary dist for OU dXt = -a* Xt * dt + sqrt(2a)*sigma*dW_t or
             diffusion coeficient for pinned brownian prior
      tfinal: terminal time value
      brown: flag for brownian reference process

  Returns:
      -(lnπ(X_T) - ln N(X_T; 0, sigma))
  """

  _, d = x_terminal.shape
  ln_target = lnpi(x_terminal)

  if brown:
    sigma = np.sqrt(tfinal) * sigma

  equi_normal = distrax.MultivariateNormalDiag(
      np.zeros(d), sigma * np.ones(d))  # equilibrium distribution

  log_ou_equilibrium = equi_normal.log_prob(x_terminal)
  lrnd = -(ln_target - log_ou_equilibrium)

  return lrnd


def relative_kl_objective(augmented_trajectory, g,
                          stl=False, trim=2, dim=2):
  """Vanilla relative KL control objective.

  Args:
      augmented_trajectory: X_{1:T} samples with ||u_t||^2/gamma as dim d+1
      g: terminal cost function typically - ln dπ/dp_1
      stl: boolean marking stl estimator usage
      trim: size of the augmented state space

  Returns:
      kl control loss
  """

  energy_cost_dt = augmented_trajectory[:, -1, -1]
  x_final_time = augmented_trajectory[:, -1, :dim]

  # import pdb; pdb.set_trace()

  stl = augmented_trajectory[:, -1, dim] if stl else 0

  terminal_cost = g(x_final_time)
  return (energy_cost_dt + terminal_cost + stl).mean()


def prob_flow_lnz(
    augmented_trajectory, eq_dist,
    target_dist, _=False, debug=False):
  """Vanilla relative KL control objective.

  Args:
      augmented_trajectory: X_{1:T} samples with trace as final dim
      eq_dist: equilibriium distribution log prob
      target_dist: log target distribution (up to Z)

  Returns:
      kl control loss
  """

  trim = 2

  trace = augmented_trajectory[:, -1, -1]
  x_init_time = augmented_trajectory[:, 0, :-trim]
  x_final_time = augmented_trajectory[:, -1, :-trim]


  ln_gamma = target_dist(x_final_time)
  lnq_0 = eq_dist(x_init_time)
  lnq = lnq_0 - trace  # Instantaneous change of variables formula
  lns = ln_gamma - lnq

  ln_numsamp = np.log(lns.shape[0])
  lnz = jscipy.special.logsumexp(lns, axis=0)  - ln_numsamp

  if debug: import pdb; pdb.set_trace()
  return -lnz


def dds_kl_objective(augmented_trajectory, *_, **__):
  """DEPRECATED DO NOT USE.

  Mostly serves as a placeholder as this is computed in the solver.

  Args:
      augmented_trajectory: tuple with trajectory and loss
      *_: empty placeholder
      **__: empty placeholder

  Returns:
      kl control loss
  """
  (_, loss) = augmented_trajectory
  return (loss).mean()


def importance_weighted_partition_estimate(augmented_trajectory, g, dim=2):
  """See TODO.

  Args:
      augmented_trajectory: X_{1:T} samples with ||u_t||^2/gamma as dim d+1
      g: terminal cost function typically - ln dπ/dp_1

  Returns:
      smoothed crosent control loss
  """

  energy_cost_dt = augmented_trajectory[:, -1, -1]

  x_final_time = augmented_trajectory[:, -1, :dim]

  stl = augmented_trajectory[:, -1, dim]

  terminal_cost = g(x_final_time)
  s_omega = -(energy_cost_dt + terminal_cost + stl)

  ln_numsamp = np.log(s_omega.shape[0])
  lnz = jscipy.special.logsumexp(s_omega, axis=0)  - ln_numsamp
  return - lnz


def importance_weighted_partition_estimate_dds(augmented_trajectory, _):
  """Logsumexp IS estimator for dds.

  Args:
      augmented_trajectory:  tuple with trajectory and dds loss

  Returns:
      smoothed crosent control loss
  """

  _, loss = augmented_trajectory

  ln_numsamp = np.log(loss.shape[0])
  lnz = jscipy.special.logsumexp(-loss, axis=0)  - ln_numsamp
  return - lnz
