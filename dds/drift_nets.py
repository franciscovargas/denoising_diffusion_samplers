"""Small AlexNet styled architecture model definition.

The number of CNN blocks, block arguments such as number of channels and kernel
shapes and strides, as well as use of dropout and batchnorm and are passed
using `ConfigDict`.
"""
from typing import Optional, Union, Callable

import haiku as hk
import jax
import jax.numpy as np


from ml_collections import config_dict as configdict


def gelu(x):
  """We use this in place of jax.nn.relu because the approximation used.

  Args:
    x: input

  Returns:
    GELU activation
  """
  return x * 0.5 * (1.0 + jax.scipy.special.erf(x / np.sqrt(2.0)))


def get_timestep_embedding(timesteps: np.array,
                           embedding_dim: np.array = 128):
  """From Fairseq.

  Build sinusoidal embeddings.
  This matches the implementation in tensor2tensor, but differs slightly
  from the description in Section 3.5 of "Attention Is All You Need".
  https://github.com/pytorch/fairseq/blob/master/fairseq/modules/sinusoidal_positional_embedding.py

  Args:
    timesteps: timesteps to embed
    embedding_dim: dimension of positional embeddings

  Returns:
    Embedded timesteps
  """
  half_dim = embedding_dim // 2
  emb = np.log(10000) / (half_dim - 1)
  emb = np.exp(np.arange(half_dim) * - emb)

  emb = np.squeeze(timesteps * emb[None, None])
  emb = np.concatenate([np.sin(emb), np.cos(emb)], axis=-1)
  if embedding_dim % 2 == 1:  # zero pad
    emb = np.pad(emb, [0, 1], [(0, 0), (0, 1)], mode="constant")

  return emb


class LinearConsInit(hk.Module):
  """Linear layer with constant init.
  """

  def __init__(self, output_size, alpha=1, name=None):
    super().__init__(name=name)
    self.alpha = alpha
    self.output_size = output_size

  def __call__(self, x):
    j, k = x.shape[-1], self.output_size
    w_init = hk.initializers.Identity(gain=self.alpha)
    w = hk.get_parameter("w", shape=[j, k], dtype=x.dtype, init=w_init)
    b = hk.get_parameter("b", shape=[k], dtype=x.dtype, init=np.zeros)

    return np.dot(x, w) + b


class LinearZero(hk.Module):
  """Linear layer with zero init.
  """

  def __init__(self, output_size, alpha=1, name=None):
    super().__init__(name=name)
    self.alpha = alpha
    self.output_size = output_size

  def __call__(self, x):
    j, k = x.shape[-1], self.output_size
    w = hk.get_parameter("w", shape=[j, k], dtype=x.dtype, init=np.zeros)
    b = hk.get_parameter("b", shape=[k], dtype=x.dtype, init=np.zeros)

    return np.dot(x, w) + b


class OUDrift(hk.Module):
  """Helper class holding OU drift.
  """
  alpha: Union[float, np.ndarray]
  sigma: Union[float, np.ndarray]
  dim: int

  def __init__(self, alpha, sigma, dim):
    super().__init__(name="ou_drift")
    self.dim = dim
    self.alpha = alpha
    self.sigma = sigma

  def __call__(self, y, t, args=None):
    return -self.alpha * y


class SimpleDriftNet(hk.Module):
  """OU Drift Net, the model is initialised to explore in early stages.

  Uses a skip connection to model an OU process at initialisation

  Attributes:
    config: ConfigDict specifying model architecture
  """

  def __init__(self,
               architecture_specs: configdict.ConfigDict,
               dim: int,
               name: Optional[str] = None):
    super(). __init__(name=name)

    self.alpha = architecture_specs.alpha
    self.architecture_specs = architecture_specs
    self.state_dim = dim
    self.dim = dim + 1
    self._grad_ln = hk.LayerNorm(-1, True, True)
    self._en_ln = hk.LayerNorm(-1, True, True)
    self.nn_clip = 1.0e3
    self.target = self.architecture_specs.target

    self.state_time_grad_net = hk.Sequential([
        hk.Sequential([hk.Linear(x), self.architecture_specs.activation])
        for x in self.architecture_specs.fully_connected_units
    ] + [LinearZero(dim)])

  def __call__(self,
               input_array: np.ndarray,
               time_array: np.ndarray,
               target: Optional[Callable[[np.ndarray], np.ndarray]] = None,
               training: Optional[bool] = True) -> np.ndarray:
    """Evaluates (carries out a forward pass) the model at train/inference time.

    Passing score information as an early input feature to network as done by
    @wgrathwohl's initiatial implementation. Found a skip connection works
    better.

    Args:
        input_array:  state to the network (N_points, N_dim)
        time_array:  time  to the network (N_points, 1)
        target: ln pi target for ULA based features
        training: if true evaluates the network in training phase else inference

    Returns:
        returns an ndarray of logits (N_points, n_dim)
    """

    time_array_emb = get_timestep_embedding(time_array)
    target = self.target
    # Using score information as a feature
    if target is not None:

      energy = target(input_array).reshape(-1, 1)
      en_plus_bias_ln = np.concatenate(
          [jax.nn.tanh(self._en_ln(energy)),
           np.ones((energy.shape[0], 1))], -1)

      grad = hk.grad(lambda _x: target(_x).sum())(input_array)
      grad = jax.lax.stop_gradient(grad)

      grad_plus_bias = np.concatenate([grad, np.ones((grad.shape[0], 1))], -1)
      grad_plus_bias_ln = self._grad_ln(grad_plus_bias)

      extended_input = np.concatenate(
          (input_array, grad_plus_bias_ln, en_plus_bias_ln, time_array_emb),
          axis=-1)
    else:
      extended_input = np.concatenate((input_array, time_array_emb), axis=-1)

    out = self.state_time_grad_net(extended_input)

    out = np.clip(
        out, -self.nn_clip, self.nn_clip
    )

    return out


class PISGRADNet(hk.Module):
  """PIS Grad network. Other than detaching should mimic the PIS Grad network.

  We detach the ULA gradients treating them as just features leading to much
  more stable training than PIS Grad.

  Attributes:
    config: ConfigDict specifying model architecture
  """

  def __init__(self,
               architecture_specs: configdict.ConfigDict,
               dim: int,
               name: Optional[str] = None):
    super(). __init__(name=name)

    self.alpha = architecture_specs.alpha
    self.stop_grad = architecture_specs.stop_grad
    self.architecture_specs = architecture_specs
    self.n_layers = len(self.architecture_specs.fully_connected_units)

    # For most PIS_GRAD experiments channels = 64
    self.channels = self.architecture_specs.fully_connected_units[0]
    self.timestep_phase = hk.get_parameter(
        "timestep_phase", shape=[1, self.channels], init=np.zeros)

    # Exact time_step coefs used in PIS GRAD
    self.timestep_coeff = np.linspace(
        start=0.1, stop=100, num=self.channels)[None]

    # This implements the time embedding for the non grad part of the network
    self.time_coder_state = hk.Sequential([
        hk.Linear(self.channels),
        self.architecture_specs.activation,
        hk.Linear(self.channels),
    ])

    # This carries out the time embedding for the NN(t) * log grad target
    self.time_coder_grad = hk.Sequential([hk.Linear(self.channels)] + [
        hk.Sequential(
            [self.architecture_specs.activation,
             hk.Linear(self.channels)]) for _ in range(self.n_layers)
    ] + [self.architecture_specs.activation,
         LinearConsInit(dim, 0)])

    # If images, do a CNN, TODO(vargfran): Could be a nice feature in future
    # if "cnn" in self.architecture_specs and self.architecture_specs.cnn:
    #   dim_rt = np.sqrt(dim)
    #   kernels = [3, 3]
    #   channels = [6, 16]
    #   activation = self.architecture_specs.activation
    #   self.state_cnn = hk.Sequential([hk.Reshape((-1, dim_rt, dim_rt))] + [
    #       hk.Sequential([hk.Conv2D(channels[i], kernels[i]), activation])
    #       for i in range(len(channels))
    #   ] + [hk.Flatten])

    # Time embedding and state concatenated network NN(x, emb(t))
    # This differs to PIS_grad where they do NN(Wx + emb(t))
    self.state_time_net = hk.Sequential([
        hk.Sequential([hk.Linear(x), self.architecture_specs.activation])
        for x in self.architecture_specs.fully_connected_units
    ] + [LinearZero(dim)])

    self.state_dim = dim
    self.dim = dim + 1
    self._grad_ln = hk.LayerNorm(-1, True, True)
    self.nn_clip = 1.0e4
    self.lgv_clip = 1.0e2

  def get_pis_timestep_embedding(self, timesteps: np.array):
    """PIS based timestep embedding.

    Args:
      timesteps: timesteps to embed

    Returns:
      embedded timesteps
    """

    sin_embed_cond = np.sin(
        (self.timestep_coeff * timesteps) + self.timestep_phase
    )
    cos_embed_cond = np.cos(
        (self.timestep_coeff * timesteps) + self.timestep_phase
    )
    return np.concatenate([sin_embed_cond, cos_embed_cond], axis=-1)

  def __call__(self,
               input_array: np.ndarray,
               time_array: np.ndarray,
               target: Optional[Callable[[np.ndarray], np.ndarray]] = None,
               training: Optional[bool] = True,
               ode: Optional[bool] = False) -> np.ndarray:
    """Evaluates (carries out a forward pass) the model at train/inference time.

    Args:
        input_array:  state to the network (N_points, N_dim)
        time_array:  time  to the network (N_points, 1)
        target: ln pi target for ULA based features
        training: if true evaluates the network in training phase else inference
        ode: Flag to turn of stopgrad for probflow estimator

    Returns:
        returns an ndarray of logits (N_points, n_dim)
    """

    time_array_emb = self.get_pis_timestep_embedding(time_array)

    grad_bool = self.stop_grad and not ode
    # Using score information as a feature
    grad = hk.grad(lambda _x: target(_x).sum())(input_array)
    grad = jax.lax.stop_gradient(grad) if grad_bool else grad
    grad = np.clip(grad, -self.lgv_clip, self.lgv_clip)

    t_net_1 = self.time_coder_state(time_array_emb)
    t_net_2 = self.time_coder_grad(time_array_emb)

    extended_input = np.concatenate((input_array, t_net_1), axis=-1)
    out_state = self.state_time_net(extended_input)

    # out_state = np.clip(
    #     out_state, -self.nn_clip, self.nn_clip
    # )

    out_state_p_grad = out_state + t_net_2 * grad
    return out_state_p_grad
