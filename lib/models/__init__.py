from .snakediff import SnakeDiffusion

import jax
import haiku as hk
from collections import namedtuple

model_type = namedtuple("Model", "get_features predict_next predict")

def get_model(*dummy_in, seed=jax.random.PRNGKey(39)):
    def f():
      model = SnakeDiffusion()
      def init(x, img, t_emb):
        img_features = model.get_features(img)
        prediction = model.predict_next(x, img_features, t_emb)
        return prediction
      return init, model_type(model.get_features, model.predict_next, model.predict)

    model = hk.multi_transform(f)
    params = model.init(seed, *jax.tree_map(lambda x: x[:1], dummy_in))

    return model, params
