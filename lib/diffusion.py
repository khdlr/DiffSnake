import jax
import jax.numpy as jnp
from . import config

def get_alpha(t):
  schedule = config.diffusion.alpha_schedule
  x = t / config.diffusion.steps_train

  if schedule == 'linear':
    return 1 - x
  elif schedule == 'circular':
    return 1 - jnp.sqrt(2*x - x*x)
  elif schedule == 'sinusoidal':
    return jnp.sin(jnp.pi / 2 * (1 - x))
  elif schedule == 'cosine':
    return 0.5 + 0.5 * jnp.cos(x * jnp.pi)
  else:
    raise ValueError(f'{config.diffusion.alpha_schedule!r}')


def sample(model, state, imgs, keys):
  ts = 

  img_features = model.get_features(state.params, imgs)

  # DDIM inference step
  def inference_step(x_t, step_vars):
    t, tm1, key = ts
    a_t, a_tm1 = get_alpha(t), get_alpha(tm1)

    eps = model.(

    x_tm1 = jnp.sqrt(a_tm1) * (x_t - jnp.sqrt(1 - a_t) * 


    return x_t, x_t
  jax.lax.scan
