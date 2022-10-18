import jax
import jax.numpy as jnp
import haiku as hk
from ... import config


class UNet1D:
  def __init__(self):
    cfg = config.model.predictor
    self.width = cfg.width
    self.depth = cfg.depth

  def __call__(self, x, t):
    skip_connections = []

    W = self.width
    channel_seq = [W * 2**i for i in range(self.depth)]
    for channels in channel_seq:
      x = Convx2(x, channels)
      skip_connections.append(x)
      x = hk.max_pool(x, 2, 2, padding='SAME')

    x = Convx2(x, 2 * channel_seq[-1])

    for channels, skip in zip(reversed(channel_seq), reversed(skip_connections)):
      B,  T,  C  = x.shape
      B_, T_, C_ = skip.shape

      upsampled = jax.image.resize(x, [B, T_, C], method='bilinear')
      x = hk.Conv1D(C_, 2, with_bias=False)(upsampled)
      x = LayerNorm()(x)
      x = jax.nn.relu(x)
      x = Convx2(jnp.concatenate([x, skip], axis=-1), channels)

    x = hk.Conv1D(2, 1, with_bias=False, w_init=jnp.zeros)(x)
    return x


def LayerNorm():
    return hk.LayerNorm(axis=-1, param_axis=-1,
                        create_scale=True, create_offset=True)


def Convx2(x, channels):
    x = hk.Conv1D(channels, 3, with_bias=False)(x)
    x = LayerNorm()(x)
    x = jax.nn.relu(x)
    x = hk.Conv1D(channels, 3, with_bias=False)(x)
    x = LayerNorm()(x)
    x = jax.nn.relu(x)
    return x
