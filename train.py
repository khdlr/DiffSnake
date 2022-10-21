import yaml
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import optax
from data_loading import get_loader 
from functools import partial
from collections import defaultdict

import wandb
from tqdm import tqdm

import sys
from munch import munchify
from lib import utils, losses, logging, models, config, diffusion
from lib.utils import TrainingState, prep, changed_state, save_state, load_state
import lib.models.nnutils as nn
from evaluate import test_step, METRICS
from einops import rearrange, repeat

from jax.config import config as jax_config
jax_config.update("jax_numpy_rank_promotion", "raise")

PATIENCE = 100

def get_optimizer():
  conf = config.opt
  lr_schedule = getattr(optax, conf.schedule_type)(**conf.schedule)
  return getattr(optax, conf.type)(lr_schedule, **conf.args)


@partial(jax.jit, static_argnums=3)
def train_step(batch, state, key, net):
  _, optimizer = get_optimizer()
  diff = config.diffusion

  aug_key, time_key, eps_key = jax.random.split(key, 3)
  img, mask, contour = prep(batch, aug_key, augment=True)
  B, H, W, C = img.shape

  t = jax.random.randint(time_key, [B, diff.snakes_per_image//2], 1, diff.steps_train)
  t = jnp.concatenate([t, diff.steps_train + 1 - t], axis=1)

  x_0 = repeat(contour, 'B T C -> B S T C', S=diff.snakes_per_image)
  alpha = diffusion.get_alpha(t)
  eps = jax.random.normal(eps_key, x_0.shape)
  x_t = jnp.clip(x_0 * jnp.sqrt(alpha) + eps * jnp.sqrt(1. - alpha), 0, 1)
  
  def get_loss(params):
    img_features = net.get_features(params, img)
    # TODO: Dropout?
    # img_features = [nn.channel_dropout(f, 0., 5) for f in img_features]

    def predict_single(x_t, t):
      return net.predict_next(params, x_t, img_features, t)

    predictions = jax.vmap(predict_single, in_axes=[1, 1], out_axes=1)(x_t, t)

    loss = jnp.mean(jnp.square(eps - predictions))

    return loss

  loss, gradients = jax.value_and_grad(get_loss)(state.params)
  updates, new_opt = optimizer(gradients, state.opt, state.params)
  new_params = optax.apply_updates(state.params, updates)

  terms = {
      'loss': loss,
      'mask': mask,
      'contour': contour,
      'imagery': img,
  }

  # Convert from normalized to to pixel coordinates

  return terms, changed_state(state,
      params=new_params,
      opt=new_opt,
  )


if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] != '-f':
        utils.assert_git_clean()
    train_key = jax.random.PRNGKey(42)
    persistent_val_key = jax.random.PRNGKey(27)

    config.update(munchify(yaml.load(open('config.yml'), Loader=yaml.SafeLoader)))

    # initialize data loading
    train_key, subkey = jax.random.split(train_key)
    B = config.batch_size
    train_loader = get_loader(B, 4, 'train', config, subkey)
    trainval_loader = get_loader(4, 1, 'train', config, None, subtiles=False)
    val_loader   = get_loader(4, 1, 'validation', config, None, subtiles=False)

    img, mask, contour = prep(next(iter(train_loader)))
    S, params = models.get_model(contour, img, jnp.zeros([img.shape[0]]))

    # Initialize model and optimizer state
    opt_init, _ = get_optimizer()
    state = TrainingState(params=params, opt=opt_init(params))
    net = S.apply

    running_min = np.inf
    last_improvement = 0
    wandb.init(project=f'Snake Diffusion', config=config)

    run_dir = Path(f'runs/{wandb.run.id}/')
    run_dir.mkdir(parents=True)
    config.run_id = wandb.run.id
    with open(run_dir / 'config.yml', 'w') as f:
        f.write(yaml.dump(config, default_flow_style=False))

    for epoch in range(1, 1001):
        wandb.log({f'epoch': epoch}, step=epoch)
        prog = tqdm(train_loader, desc=f'Ep {epoch} Trn')
        trn_metrics = defaultdict(list)
        loss_ary = None
        for step, batch in enumerate(prog, 1):
            train_key, subkey = jax.random.split(train_key)
            terms, state = train_step(batch, state, subkey, net)
            trn_metrics['loss'].append(terms['loss'])

        logging.log_metrics(trn_metrics, 'trn', epoch, do_print=False)

        if epoch % 10 != 0:
             continue

        # Save Checkpoint
        save_state(state, run_dir / f'latest.pkl')

        # Validate
        for mode, loader in [('trainval', trainval_loader), ('val', val_loader)]:
          val_key = persistent_val_key
          val_metrics = defaultdict(list)
          for step, batch in enumerate(tqdm(loader)):
              val_key, subkey = jax.random.split(val_key)
              metrics, out = test_step(batch, state, subkey, net)

              for m in metrics:
                val_metrics[m].append(metrics[m])

              out = jax.tree_map(lambda x: x[0], out) # Select first example from batch
              logging.log_anim(out, f"Animated_{mode}/{step}", epoch)

              if mode == 'trainval' and step >= 8:
                break

          logging.log_metrics(val_metrics, mode, epoch)
