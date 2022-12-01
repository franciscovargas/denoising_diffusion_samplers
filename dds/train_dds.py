"""Main training file.

For training diffusion based samplers (OU reversal SDE and Follmer SDE )
"""
import functools
import timeit
from typing import Any, List, Tuple
from absl import flags

from absl import logging
import haiku as hk
import jax
import jax.numpy as jnp

from ml_collections import config_dict as configdict
from ml_collections import config_flags

import numpy as onp
import optax

from jaxline import utils

from dds.configs.config import set_task
from dds.data_paths import results_path


FLAGS = flags.FLAGS
Writer = Any

# lr_sonar, funnel, lgcp, ion, nice, vae, brownian
_TASK = flags.DEFINE_string("task", "lr_sonar", "Inference task name.")
config_flags.DEFINE_config_file(
    "config",
    "//dds/config.py",
    lock_config=False,
    help_string="Path to ConfigDict file."
)


class WandbWriterWrapper:

  def __init__(self, *args, **kwargs):
    pass

  def write(self, *args):
    pass


def update_detached_params(trainable_params, non_trainable_params,
                           attached_network_name="simple_drift_net",
                           detached_network_name="stl_detach"):
  """Auxiliary function updating detached params for STL.

  Args:
      trainable_params:
      non_trainable_params:
      attached_network_name:
      detached_network_name:
  Returns:
    Returns non trainable params
  """

  if len(trainable_params) != len(non_trainable_params):
    return non_trainable_params

  for key in trainable_params.keys():
    if attached_network_name in key:
      key_det = key.replace(attached_network_name, detached_network_name)
    else:
      key_det = key.replace("diffusion_network",
                            detached_network_name + "_diff")
    non_trainable_params[key_det] = trainable_params[key]  # pytype: disable=unsupported-operands

  return non_trainable_params


def train_dds(
    config: configdict.ConfigDict):
# ) -> Tuple[hk.Params, hk.State, hk.TransformedWithState, jnp.ndarray,
#            List[float]]:
  """Train Follmer SDE.

  Args:
    config : ConfigDict with model and training parameters.

  Returns:
    Tuple containing params, state, function that runs learned sde, and losses
  """

  # train setup
  data_dim = config.model.input_dim
  device_no = jax.device_count()

  alpha = config.model.alpha
  sigma = config.model.sigma
  m = config.model.m

  batch_size_ = int(config.model.batch_size / device_no)
  batch_size_elbo = int(config.model.elbo_batch_size / device_no)

  step_scheme = config.model.step_scheme_dict[config.model.step_scheme_key]

  dt = config.model.dt

  if config.model.reference_process_key == "oududp":
    key_conversion = {
        "pis": "pisudp",
        "vanilla": "vanilla_udp",
        "tmpis": "tmpis_udp"
    }
    # "pisudp"
    config.model.network_key = key_conversion[config.model.network_key]

  net_key = config.model.network_key
  network = config.model.network_dict[net_key]

  tpu = config.model.tpu

  detach_dif_path, detach_dritf_path = (
      config.model.detach_path, config.model.detach_path)

  target = config.model.target

  tfinal = config.model.tfinal
  lnpi = config.trainer.lnpi

  ref_proc_key = config.model.reference_process_key
  ref_proc = config.model.reference_process_dict[ref_proc_key]

  trim = (2 if "stl" in str(ref_proc).lower() or "udp" in str(ref_proc).lower()
          else 1)
 
  stl = config.model.stl

  brown = "brown" in str(ref_proc).lower()

  seed = config.trainer.random_seed  if "random_seed" in config.trainer else 42

  # task directory (currently not in use)
  task = config.task
  method = config.model.reference_process_key
  task_path = results_path + f"/{task}" + f"/{ref_proc_key}" + f"/{net_key}"
  task_path += f"/{method}"


  # checkpoiting variables for wandb
  nsteps = config.model.ts.shape[0]
  keep_every_nth = int(config.trainer.epochs / 125)
  file_name = (f"/alpha_{alpha}_sigma_{sigma}_epochs_{config.trainer.epochs}" +
               f"_task_{task}_seed_{seed}_steps_{nsteps}_stl_{stl}_{method}" +
               f"_scheme_{config.model.step_scheme_key}_ddpm_test11_chk")
  _ = task_path + file_name

  detach_stl_drift = (
      config.model.detach_stl_drift if
      "detach_stl_drift" in config.model else False
  )

  drift_network = lambda: network(config.model, data_dim, "simple_drift_net")

  ############## wandb logging  place holder ################
  data_id = "denoising_diffusion_samplers"  # Project name
  training_writer = WandbWriterWrapper(data_id, dataframe="elbo_results")
  training_writer_eval = WandbWriterWrapper(data_id, dataframe="elbo_results_eval")
  is_writer = WandbWriterWrapper(data_id, dataframe="is_results")
  is_writer_eval = WandbWriterWrapper(data_id, dataframe="is_results_eval")
  pf_writer = WandbWriterWrapper(data_id, dataframe="pf_results")
  pf_writer_eval = WandbWriterWrapper(data_id, dataframe="pf_results_eval")
  lr_writer = WandbWriterWrapper(data_id, dataframe="lr")

  def _forward_fn(batch_size: int,
                  training: bool = True,
                  ode=False, dt_=dt) -> jnp.ndarray:

    model_def = ref_proc(
        sigma, data_dim, drift_network, tfinal=tfinal, dt=dt_,
        step_scheme=step_scheme, alpha=alpha, target=target, tpu=tpu,
        detach_stl_drift=detach_stl_drift, diff_net=None,
        detach_dritf_path=detach_dritf_path, detach_dif_path=detach_dif_path,
        m=m, log=config.model.log, exp_bool=config.model.exp_dds
    )

    return model_def(batch_size, training, ode=ode)

  forward_fn = hk.transform_with_state(_forward_fn)

  # opt and loss setup
  seq = hk.PRNGSequence(seed)
  rng_key = next(seq)
  # subkeys = jax.random.split(rng_key, device_no)
  subkeys = utils.bcast_local_devices(rng_key)

  p_init = jax.pmap(
      functools.partial(forward_fn.init, batch_size=batch_size_,
                        training=True), axis_name="num_devices")

  params, model_state = p_init(subkeys)

  trainable_params, non_trainable_params = hk.data_structures.partition(
      lambda module, name, value: "stl_detach" not in module, params)

  clipper = optax.clip(1.0)
  base_dec = config.trainer.lr_sch_base_dec
  scale_by_adam = optax.scale_by_adam()
  # if base_dec == 0:
  #   scale_by_lr = optax.scale(-config.trainer.learning_rate)
  #   opt = optax.chain(clipper, scale_by_adam, scale_by_lr)
  # else:
  transition_steps = 50
  exp_lr = optax.exponential_decay(config.trainer.learning_rate,
                                   transition_steps, base_dec)
  scale_lr = optax.scale_by_schedule(exp_lr)
  opt = optax.chain(clipper, scale_by_adam, scale_lr, optax.scale(-1))

  # opt = optax.adam(learning_rate=config.trainer.learning_rate)
  opt_state = jax.pmap(opt.init)(trainable_params)

  @functools.partial(
      jax.pmap, axis_name="num_devices", static_broadcasted_argnums=(3, 4, 5))
  def forward_fn_jit(
      params,
      model_state: hk.State,
      subkeys: jnp.ndarray,
      batch_size: jnp.ndarray, ode=False, dt_=dt):

    samps, _ = forward_fn.apply(
        params,
        model_state,
        subkeys,
        int(batch_size / device_no),
        False,
        ode=ode, dt_=dt_)
    samps = jax.device_get(samps)

    augmented_trajectory, ts = samps
    return (augmented_trajectory, ts), _

  def forward_fn_wrap(
      params,
      model_state: hk.State,
      rng_key: jnp.ndarray,
      batch_size: jnp.ndarray, ode=False, dt_=dt):
    subkeys = jax.random.split(rng_key, device_no)
    (augmented_trajectory, ts), _ = forward_fn_jit(params, model_state,
                                                   subkeys, batch_size, ode,
                                                   dt_)

    dv, ns, t, _ = augmented_trajectory.shape
    augmented_trajectory = augmented_trajectory.reshape(dv*ns, t, -1)
    return (augmented_trajectory, utils.get_first(ts)), _

  def full_objective(
      trainable_params,
      non_trainable_params,
      model_state: hk.State,
      rng_key: jnp.ndarray,
      batch_size: int,
      is_training: bool = True,
      ode: bool = False,
      stl: bool = False,
    ):

    params = hk.data_structures.merge(trainable_params, non_trainable_params)
    (augmented_trajectory, _), model_state = forward_fn.apply(
        params, model_state, rng_key, batch_size, True, ode
    )

    # import pdb; pdb.set_trace()
    gpartial = functools.partial(
        config.model.terminal_cost,
        lnpi=lnpi, sigma=sigma, tfinal=tfinal, brown=brown)
    
    if is_training:
      loss = config.trainer.objective(
          augmented_trajectory, gpartial, stl=stl, trim=trim, dim=data_dim)
    elif not ode:
      loss = config.trainer.lnz_is_estimator(
          augmented_trajectory, gpartial, dim=data_dim)
    else:
      loss = config.trainer.lnz_pf_estimator(
          augmented_trajectory, config.model.source, config.model.target)
    return loss, model_state

  @functools.partial(
      jax.pmap, axis_name="num_devices", static_broadcasted_argnums=(5,))
  def update(
      trainable_params,
      non_trainable_params,
      model_state: hk.State,
      opt_state: Any,
      rng_key: jnp.ndarray,
      batch_size: jnp.ndarray):
    grads, new_model_state = jax.grad(
        full_objective, has_aux=True)(
            trainable_params,
            non_trainable_params,
            model_state,
            rng_key,
            batch_size,
            is_training=True,
            stl=stl)
    grads = jax.lax.pmean(grads, axis_name="num_devices")

    updates, opt_state = opt.update(grads, opt_state)
    new_params = optax.apply_updates(trainable_params, updates)
    return new_params, opt_state, new_model_state

  @functools.partial(
      jax.pmap, axis_name="num_devices", static_broadcasted_argnums=(4, 5, 6))
  def jited_val_loss(
      trainable_params,
      non_trainable_params,
      model_state: hk.State,
      rng_key: jnp.ndarray,
      batch_size: jnp.ndarray,
      is_training: bool = True,
      ode: bool = False):

    loss, new_model_state = full_objective(
        trainable_params,
        non_trainable_params,
        model_state,
        rng_key,
        batch_size,
        is_training=is_training, ode=ode,
        stl=False)

    loss = jax.lax.pmean(loss, axis_name="num_devices")
    return loss, new_model_state

  def eval_report(
      trainable_params,
      non_trainable_params,
      model_state: hk.State,
      rng_key: jnp.ndarray,
      batch_size: int,
      epoch: int,
      writer: Writer,
      loss_list: List[float],
      is_training: bool = True,
      print_flag: bool = False,
      ode: bool = False,
  ) -> None:

    loss, model_state = jited_val_loss(
        trainable_params, non_trainable_params,
        model_state, rng_key, batch_size, is_training, ode)
    loss = jax.device_get(loss)
    loss = onp.asarray(utils.get_first(loss).item()).item()

    log_string = "epoch: %s %s  loss: %s", epoch, "TRAIN", loss
    logging.info(log_string)
    if config.trainer.notebook and print_flag: print(log_string)

    loss_list.append(loss)
    writer.write({"epoch": epoch, "loss": loss})
    # writer.flush()

  loss_list = []
  loss_list_is = []
  loss_list_pf = []

  start = 0
  times = []
  for epoch in range(start, config.trainer.epochs):
    rng_key = next(seq)
    subkeys = jax.random.split(rng_key, device_no)

    trainable_params, opt_state, model_state = update(trainable_params,
                                                      non_trainable_params,
                                                      model_state, opt_state,
                                                      subkeys, batch_size_)
    if config.trainer.timer:
      def func():
        return jax.block_until_ready(
            update(trainable_params, non_trainable_params, model_state,
                   opt_state, subkeys, batch_size_))

      delta_time = timeit.timeit(func, number=1)
      times.append(delta_time)

    update_detached_params(trainable_params, non_trainable_params,
                           "simple_drift_net", "stl_detach")

    if epoch % config.trainer.log_every_n_epochs == 0:

      eval_report(trainable_params, non_trainable_params,
                  model_state, subkeys, batch_size_elbo, epoch,
                  training_writer, loss_list, print_flag=True)

      eval_report(trainable_params, non_trainable_params,
                  model_state, subkeys, batch_size_elbo, epoch,
                  is_writer, loss_list_is, is_training=False)

      eval_report(trainable_params, non_trainable_params,
                  model_state, subkeys, batch_size_elbo, epoch,
                  pf_writer, loss_list_pf, is_training=False, ode=True)

      lr = onp.asarray(exp_lr(epoch).item()).item()
      lr_writer.write({"epoch": epoch, "lr": lr})

  loss_list_is_eval, loss_list_eval, loss_list_pf_eval = [], [], []
  for i in range(config.eval.seeds):
    rng_key = next(seq)
    subkeys = jax.random.split(rng_key, device_no)
    eval_report(
        trainable_params,
        non_trainable_params,
        model_state,
        subkeys,
        batch_size_elbo,
        i,
        training_writer_eval,
        loss_list_eval,
        print_flag=True)

    eval_report(
        trainable_params,
        non_trainable_params,
        model_state,
        subkeys,
        batch_size_elbo,
        i,
        is_writer_eval,
        loss_list_is_eval,
        is_training=False)

    eval_report(
        trainable_params,
        non_trainable_params,
        model_state,
        subkeys,
        batch_size_elbo,
        i,
        pf_writer_eval,
        loss_list_pf_eval,
        is_training=False, ode=True)

  params = hk.data_structures.merge(trainable_params, non_trainable_params)
  if config.trainer.timer:
    print(times[1:])

  samps = 2500
  if method == "lgcp" and tfinal >= 12:
    samps = 100

  (augmented_trajectory, _), _ = forward_fn_wrap(params, model_state, rng_key,
                                                 samps)

  (augmented_trajectory_det, _), _ = forward_fn_wrap(params, model_state,
                                                     rng_key, samps, True)


  results_dict = {
      "elbo": loss_list,
      "is": loss_list_is,
      "pf": loss_list_pf,
      "elbo_eval": loss_list_eval,
      "is_eval": loss_list_is_eval,
      "pf_eval": loss_list_pf_eval,
  }
  return params, model_state, forward_fn_wrap, rng_key, results_dict


def main(_):

  config_file = FLAGS.config
  config_file = set_task(config_file, task=_TASK.value)
  logging.info(config_file)
  train_pmap(config_file)


if __name__ == "__main__":
  pass
