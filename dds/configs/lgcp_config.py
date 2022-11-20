"""Config for LGCP model.
"""
import distrax

from jax import numpy as np

from ml_collections import config_dict as configdict

from annealed_flow_transport import densities as aft_densities

from dds import data_paths


def make_config(base_config):
  """Base config for log reg experiment.

  Args:
    base_config: configdict
  Returns:
    base config dict for experiment
  """

  base_config.task = "lgcp"

  final_config = configdict.ConfigDict()
  final_config.density = "LogGaussianCoxPines"
  final_config.use_whitened = False
  final_config.file_path = data_paths.fpines_path
  base_config.final_config_lgcp = final_config

  base_config.model.input_dim = 1600
  base_density = aft_densities.LogGaussianCoxPines(
      base_config.final_config_lgcp, num_dim=base_config.model.input_dim)
  log_prob = base_density.evaluate_log_density
  base_config.trainer.lnpi = log_prob
  base_config.model.target = log_prob

  base_config.model.elbo_batch_size = 2000

  base_config.model.source = distrax.MultivariateNormalDiag(
      np.zeros(base_config.model.input_dim),
      base_config.model.sigma * np.ones(base_config.model.input_dim)).log_prob

  base_config.model.ts = base_config.model.step_scheme(
      0, base_config.model.tfinal, base_config.model.dt, dtype=np.float32)

  return base_config
