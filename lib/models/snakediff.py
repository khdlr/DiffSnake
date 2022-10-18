import math
import jax
import jax.numpy as jnp
import haiku as hk
from .. import config

from . import backbones, predictors
from . import nnutils as nn
from einops import reduce
from .snake_utils import sample_at_vertices



class SnakeDiffusion(hk.Module):
  def __init__(self):
    super().__init__()
    cfg = config.model
    self.backbone = getattr(backbones, cfg.backbone.type)()
    self.predictor = getattr(predictors, cfg.predictor.type)()
    self.t_emb_dim = cfg.time_embedding_dim

  def get_features(self, imagery):
    return self.backbone(imagery)

  def predict_next(self, x_t, feature_maps, t):
    features = [x_t]
    t_emb = timestep_embedding(t, self.t_emb_dim)
    for feature_map in feature_maps:
        features.append(jax.vmap(sample_at_vertices, [0, 0])(x_t, feature_map))
    input_features = jnp.concatenate(features, axis=-1)

    return self.predictor(input_features, t_emb)


def timestep_embedding(t, embedding_dim):
    """
    # Ported from https://github.com/ermongroup/ddim/blob/main/models/diffusion.py

    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """

    half_dim = embedding_dim // 2
    factors = jnp.arange(half_dim)
    factors *= -math.log(10000) / (half_dim-1) 

    theta = jnp.exp(jnp.einsum('c,...->...c', factors, t))
    emb = jnp.concatenate([jnp.sin(theta), jnp.cos(theta)], axis=-1)
    return emb

