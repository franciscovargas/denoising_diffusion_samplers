"""Config for VAE model.
"""
import distrax

from jax import numpy as np

from ml_collections import config_dict as configdict

from annealed_flow_transport import densities as aft_densities

from dds.data_paths import vae_path

def make_config(base_config):
  """Base config for log reg experiment.

  Args:
    base_config: configdict
  Returns:
    base config dict for experiment
  """

  base_config.task = "vae"

  base_config.model.input_dim = 30
  final_config = configdict.ConfigDict()
  final_config.density = "AutoEncoderLikelihood"
  final_config.image_index = 3689
  final_config.params_filename = vae_path
  base_config.final_config_vae = final_config
  base_density = aft_densities.AutoEncoderLikelihood(
      base_config.final_config_vae, num_dim=base_config.model.input_dim)

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
