import jax
import jax.numpy as jnp
import haiku as hk
from einops import rearrange
from ... import config


class UNet1D:
  def __init__(self):
    cfg = config.model.predictor
    self.channel_seq = cfg.channel_seq
    self.attn_channels = cfg.attn_channels

  def __call__(self, x, t_emb):

    inout = list(zip(self.channel_seq, self.channel_seq[1:]))
    x = hk.Conv1D(self.channel_seq[0], 5, padding='SAME', with_bias=False)(x)

    skip_connections = [x]
    for c_in, c_out in inout:
      x = ResBlock(x, t_emb, c_in)
      x = ResBlock(x, t_emb, c_in)
      if c_in in self.attn_channels:
        x = x + MHA(c_in)(x, x, x)

      x = hk.Conv1D(c_out, 2, 2, padding='SAME')(x)
      skip_connections.append(x)

    jax.tree_map(lambda x: x.shape, skip_connections)

    C = self.channel_seq[-1]
    x = ResBlock(x, t_emb, C)
    x = x + MHA(C)(x, x, x)
    x = ResBlock(x, t_emb, C)

    for (c_out, _), skip in zip(reversed(inout), reversed(skip_connections)):
      x = ResBlock(x + skip, t_emb, c_out)
      x = ResBlock(x, t_emb, c_out)
      if c_out in self.attn_channels:
        x = x + MHA(c_out)(x, x, x)
      x = LayerNorm()(x)
      x = jax.nn.silu(x)
      x = hk.Conv1DTranspose(c_out, 2, 2, with_bias=False)(x)

    x = hk.Conv1D(2, 1, with_bias=False, w_init=jnp.zeros)(x)
    return x


def LayerNorm():
    return hk.LayerNorm(axis=-1, param_axis=-1,
                        create_scale=True, create_offset=True)


def ResBlock(x, t_emb, channels):
  if x.shape[-1] == channels:
    skip = x
  else:
    skip = hk.Linear(channels)(x)

  t_emb = rearrange(hk.Linear(channels, with_bias=False)(t_emb), 'B C -> B 1 C')

  x = LayerNorm()(x)
  x = jax.nn.silu(x)
  x = hk.Conv1D(channels, 3, with_bias=False)(x)

  x = x + t_emb
  
  x = LayerNorm()(x)
  x = jax.nn.silu(x)
  x = hk.Conv1D(channels, 3, with_bias=False)(x)

  return x + skip


def MHA(channels):
  n_heads = channels // 64
  return hk.MultiHeadAttention(n_heads, channels // n_heads, w_init=hk.initializers.TruncatedNormal())
