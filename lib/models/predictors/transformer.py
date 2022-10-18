# An (almost faithful) port of
# https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/transformer.py
import jax
import jax.numpy as jnp
import haiku as hk
from einops import rearrange, repeat


class TransformerEncoder(hk.Module):
    def __init__(self, num_layers, layer_type, layer_args, shared_weights):
        super().__init__()
        self.num_layers = num_layers
        self.layer_type = layer_type
        self.layer_args = layer_args
        self.shared_weights = shared_weights

    def __call__(self, x, is_training, return_all=False):
        """Params:
        x: B x T x C
        """
        if self.shared_weights:
          _layer = self.layer_type(**self.layer_args)
          layer = lambda x, is_training: _layer(x, is_training)
        else:
          layer = lambda x, is_training: self.layer_type(**self.layer_args)(x, is_training)

        results = [x]
        for _ in range(self.num_layers):
            results.append(layer(results[-1], is_training))

        if return_all:
          return results[-1], results
        else:
          return results[-1]


class TransformerDecoder(hk.Module):
    def __init__(self, num_layers, layer_type, layer_args, shared_weights):
        super().__init__()
        self.num_layers = num_layers
        self.layer_type = layer_type
        self.layer_args = layer_args
        self.shared_weights = shared_weights

    def __call__(self, target, source, is_training, return_all=False):
        """Params:
        x: B x T x C
        """
        if self.shared_weights:
          _layer = self.layer_type(**self.layer_args)
          layer = lambda x, source, is_training: _layer(x, source, is_training)
        else:
          layer = lambda x, source, is_training: \
              self.layer_type(**self.layer_args)(x, source, is_training)

        results = [target]
        for _ in range(self.num_layers):
            results.append(layer(results[-1], source, is_training))

        if return_all:
          return results[-1], results
        else:
          return results[-1]


class EncoderLayer(hk.Module):
    def __init__(self, d_model, nheads, dim_feedforward, dropout=0.1):
        super().__init__()
        self.dropout = dropout

        self.self_attn = hk.MultiHeadAttention(nheads, d_model//nheads, 1.0)
        self.linear1 = hk.Linear(dim_feedforward)
        self.linear2 = hk.Linear(d_model)

        self.norm1 = hk.LayerNorm(-1, True, True)
        self.norm2 = hk.LayerNorm(-1, True, True)

    def __call__(self, x, is_training):
        """Params:
        x: B x T x C
        """
        # Self-Attention Block
        resid = x
        x = self.self_attn(x, x, x)
        if is_training:
            x = hk.dropout(hk.next_rng_key(), self.dropout, x)
        x = self.norm1(x)
        x = resid + x

        # Feedforward-Block
        resid = x
        x = jax.nn.relu(self.linear1(x))
        if is_training:
            x = hk.dropout(hk.next_rng_key(), self.dropout, x)
        x = self.linear2(x)
        if is_training:
            x = hk.dropout(hk.next_rng_key(), self.dropout, x)
        x = self.norm2(x)
        x = resid + x

        return x


class DecoderLayer(hk.Module):
    def __init__(self, d_model, nheads, dim_feedforward, dropout=0.1):
        super().__init__()
        self.dropout = dropout

        self.self_attn  = hk.MultiHeadAttention(nheads, d_model//nheads, 1.0)
        self.cross_attn = hk.MultiHeadAttention(nheads, d_model//nheads, 1.0)
        self.linear1 = hk.Linear(dim_feedforward)
        self.linear2 = hk.Linear(d_model)

        self.norm1 = hk.LayerNorm(-1, True, True)
        self.norm2 = hk.LayerNorm(-1, True, True)
        self.norm3 = hk.LayerNorm(-1, True, True)

    def __call__(self, target, source, is_training):
        """Params:
        target: B x T x C
        source: B x N x C
        """

        # Self-Attention Block
        x = target
        resid = x
        x = self.self_attn(x, x, x)
        if is_training:
            x = hk.dropout(hk.next_rng_key(), self.dropout, x)
        x = self.norm1(x)
        x = resid + x

        # Cross-Attention Block
        resid = x
        x = self.cross_attn(x, source, source)
        if is_training:
            x = hk.dropout(hk.next_rng_key(), self.dropout, x)
        x = self.norm2(x)
        x = resid + x

        # Feedforward-Block
        resid = x
        x = jax.nn.relu(self.linear1(x))
        if is_training:
            x = hk.dropout(hk.next_rng_key(), self.dropout, x)
        x = self.linear2(x)
        if is_training:
            x = hk.dropout(hk.next_rng_key(), self.dropout, x)
        x = self.norm3(x)
        x = resid + x

        return x


class Transformer(hk.Module):
    def __init__(self, d_model, nheads, encoder_layers=6, decoder_layers=6, 
            dim_feedforward=2048, dropout=0.1, encoder_shared=False, decoder_shared=False,
            encoder_layer_type=EncoderLayer, decoder_layer_type=DecoderLayer
            ):
        super().__init__()
        self.encoder = TransformerEncoder(encoder_layers,
            layer_type=encoder_layer_type,
            layer_args=dict(
              d_model=d_model, nheads=nheads,
              dim_feedforward=dim_feedforward, dropout=dropout,
            ),
            shared_weights=encoder_shared,
        )
        self.decoder = TransformerDecoder(decoder_layers,
            layer_type=decoder_layer_type,
            layer_args=dict(
              d_model=d_model, nheads=nheads,
              dim_feedforward=dim_feedforward, dropout=dropout,
            ),
            shared_weights=decoder_shared,
        )

    def __call__(self, source, target, is_training, return_all=False):
      if return_all:
        source, enc_res = self.encoder(source, is_training, return_all=True)
        out, dec_res = self.decoder(target, source, is_training, return_all=True)
        return out, {'encoder': enc_res, 'decoder': dec_res}
      else:
        source = self.encoder(source, is_training, return_all=False)
        return self.decoder(target, source, is_training, return_all=False)


def trig_encoding(n_tokens, dim, f0=0.3, f1=16):
  angles = rearrange(jnp.linspace(0, 2*jnp.pi, n_tokens), 'T -> T 1')
  scales = jnp.logspace(jnp.log10(f0), jnp.log10(f1), dim // 2)
  scales = rearrange(scales, 'D -> 1 D')
  cos = jnp.cos(angles * scales)
  sin = jnp.sin(angles * scales)
  return jnp.concatenate([cos, sin], axis=-1)


def trig_encoding_2d(H, W, dim, f0=0.3, f1=16):
  encoding_y = repeat(trig_encoding(W, dim // 2, f0, f1), 'H D -> H W D', H=H, W=W) 
  encoding_x = repeat(trig_encoding(H, dim // 2, f0, f1), 'W D -> H W D', H=H, W=W)
  return jnp.concatenate([encoding_y, encoding_x], axis=-1)
