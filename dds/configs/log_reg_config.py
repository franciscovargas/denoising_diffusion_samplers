"""Config for log reg model.
"""
import distrax

from jax import numpy as np
from dds.targets import lr_sonar


def make_log_reg_config(base_config):
  """Base config for log reg experiment.

  Args:
    base_config: configdict
  Returns:
    base config dict for experiment
  """
  name = {
      "lr_sonar": "sonar_full.pkl",
      "ion": "ionosphere_full.pkl"
  }[base_config.task]
  log_prob, dim = lr_sonar.load_target(name)

  base_config.dataset_name = name
  base_config.trainer.lnpi = log_prob
  base_config.model.target = log_prob

  base_config.model.input_dim = dim

  base_config.model.elbo_batch_size = 300

  base_config.model.source = distrax.MultivariateNormalDiag(
      np.zeros(base_config.model.input_dim),
      base_config.model.sigma * np.ones(base_config.model.input_dim)).log_prob

  base_config.model.ts = base_config.model.step_scheme(
      0, base_config.model.tfinal, base_config.model.dt, dtype=np.float32)

  return base_config
