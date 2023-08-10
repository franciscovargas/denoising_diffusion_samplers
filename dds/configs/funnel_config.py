"""Config for Funnel experiments.
"""
import distrax

from jax import numpy as np

from ml_collections import config_dict as configdict

from dds.targets import aft_densities

from dds.configs.config import set_task, get_config


def make_config():
    """Base config for log reg experiment.

    Args:
      base_config: configdict
    Returns:
      base config dict for experiment
    """
    base_config = get_config()

    # Time and step settings (Need to be done before calling set_task)
    base_config.model.tfinal = 6.4
    base_config.model.dt = 0.05

    if base_config.model.reference_process_key == "oudstl":
        base_config.model.step_scheme_key = "cos_sq"

    base_config = set_task(base_config, "funnel")
    base_config.model.reference_process_key = "oudstl"

    if base_config.model.reference_process_key == "oudstl":
        base_config.model.step_scheme_key = "cos_sq"
        
        # Opt setting for funnel
        base_config.model.sigma = 1.075
        base_config.model.alpha = 0.6875
        base_config.model.m = 1.0
            
        # Path opt settings    
        base_config.model.exp_dds = False


    base_config.model.stl = False
    base_config.model.detach_stl_drift = False

    base_config.trainer.notebook = True
    base_config.trainer.epochs = 11000
    # Opt settings we use
    # funnel_config.trainer.learning_rate = 0.0001
    base_config.trainer.learning_rate = 5 * 10**(-3)
    base_config.trainer.lr_sch_base_dec = 0.95 # For funnel

    return base_config