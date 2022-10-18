import jax
import jax.numpy as jnp
import haiku as hk
from .. import config

from . import backbones, predictors
from . import nnutils as nn
from .snake_utils import sample_at_vertices


class SnakeDiffusion(hk.Module):
  def __init__(self):
    super().__init__()
    cfg = config.model
    self.backbone = getattr(backbones, cfg.backbone.type)()
    self.predictor = getattr(predictors, cfg.predictor.type)()

  def get_features(self, imagery):
    return self.backbone(imagery)

  def predict_next(self, x_t, feature_maps, t):
    features = [x_t]
    for feature_map in feature_maps:
        features.append(jax.vmap(sample_at_vertices, [0, 0])(x_t, feature_map))
    input_features = jnp.concatenate(features, axis=-1)

    return self.predictor(input_features, t)

  def predict(self, x_T, feature_maps, timesteps):
    features = [x_t]
    for feature_map in feature_maps:
        features.append(jax.vmap(sample_at_vertices, [0, 0])(x_t, feature_map))
    input_features = jnp.concatenate(features, axis=-1)

    return self.predictor(input_features)
