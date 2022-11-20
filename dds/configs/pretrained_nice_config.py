"""Anneal from isotropic multivariate normal to pretrained flow."""
import distrax
from jax import numpy as np
import ml_collections

from dds.targets import aft_densities


ConfigDict = ml_collections.ConfigDict


def get_config_nice() -> ConfigDict:
  """Returns a normalizing flow experiment config as ConfigDict."""
  config = ConfigDict()
  config.seed = 42
  config.use_remat = False

  diffusion = ConfigDict()
  diffusion.num_steps = 500
  diffusion.simulation_time = 10.
  network = ConfigDict()
  network.h_dim = 256
  network.n_layers = 3
  network.emb_dim = 16
  diffusion.network = network
  diffusion.loss_type = "variational"
  diffusion.warm_start = True
  config.diffusion = diffusion

  training = ConfigDict()
  training.num_iterations_training = int(1e5)
  training.batch_size = 128
  training.learning_rate = 0.001
  config.training = training


  exp_no = 9
  checkpoint_base = ""
  checkpoint_base += f""
  initial_distribution = ConfigDict()
  initial_distribution.name = "PretrainedMultivariateNormal"
  # set mean/std to empirical mean/std over data
  initial_distribution.std_padding = .1
  initial_distribution.ckpt_path = checkpoint_base
  config.initial_distribution = initial_distribution

  target_distribution = ConfigDict()
  target_distribution.name = "PretrainedNICE"
  # 28x28
  config.num_dim = 14 ** 2
  target_distribution.ckpt_path = checkpoint_base  # checkpoint_base.format(2, 2)
  config.target_distribution = target_distribution

  evaluation = ConfigDict()
  evaluation.batch_size = 128
  evaluation.num_mc_estimates = 32
  config.evaluation = evaluation

  logging = ConfigDict()
  logging.log_every = 100
  logging.eval_every = 1000
  logging.checkpoint_every = 100
  config.logging = logging

  config.hidden_dim = network.h_dim

  return config


def make_config(base_config):
  """Base config for log reg experiment.

  Args:
    base_config: configdict
  Returns:
    base config dict for experiment
  """

  base_config.task = "nice"

  base_config.final_config_nice = get_config_nice()

  base_config.model.input_dim = base_config.final_config_nice.num_dim
  base_density = aft_densities.NICE(
      base_config.final_config_nice.target_distribution,
      num_dim=base_config.final_config_nice.num_dim)

  log_prob = base_density.evaluate_log_density
  base_config.trainer.lnpi = log_prob
  base_config.model.target = log_prob
  base_config.model.nice = base_density

  base_config.model.batch_size = 600

  base_config.model.elbo_batch_size = 2000

  base_config.model.source = distrax.MultivariateNormalDiag(
      np.zeros(base_config.model.input_dim),
      base_config.model.sigma * np.ones(base_config.model.input_dim)).log_prob

  base_config.model.ts = base_config.model.step_scheme(
      0, base_config.model.tfinal, base_config.model.dt, dtype=np.float32)

  return base_config

