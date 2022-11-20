"""Base routine for hutchinsons trace estimator.
"""
import jax
import jax.numpy as jnp


def get_div_fn(fn, step_rng, shape, exact=False):
  """Create the divergence function of `fn` using the Hutchinson-Skilling trace estimator."""
  _, dim = shape
  args = {
      "ode": True
  }
  if exact:
    def div_fn_1(f):
      jac = jax.jacrev(f, 0)

      def out(x_, t_):
        full_jac = jnp.squeeze(jac(x_[None], t_, args))
        func_jac = full_jac[:dim, :dim]
        return jnp.trace(func_jac)

      return out
    def div_fn(x, t):
      div_1 = lambda data: div_fn_1(fn)(data, t)
      div_fn_ = jnp.squeeze(jax.vmap(div_1)(x))
      return div_fn_
  else:
    eps = jax.random.normal(step_rng, shape)
    def div_fn(x, t):
      # grad_fn = lambda data: jnp.sum(fn(data, t, args)[:, :dim] * eps)
      # grad_fn_eps = jax.grad(grad_fn)(x)
      grad_fn = lambda data: fn(data, t, args)[:, :dim]
      _, grad_fn_eps = jax.jvp(grad_fn, (x[:, :dim],), (eps,))

      return jnp.sum(
          grad_fn_eps[:, :dim] * eps, axis=-1)

  return div_fn
