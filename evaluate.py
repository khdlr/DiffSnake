import sys
import yaml
from functools import partial
from pathlib import Path
import jax
import jax.numpy as jnp
import wandb
from tqdm import tqdm

import models
from data_loading import get_loader
from lib import metrics, utils, logging
from lib.utils import TrainingState, prep, load_state


METRICS = dict(
    mae            = metrics.mae,
    rmse           = metrics.rmse,
    forward_mae    = metrics.forward_mae,
    backward_mae   = metrics.backward_mae,
    forward_rmse   = metrics.forward_rmse,
    backward_rmse  = metrics.backward_rmse,
    symmetric_mae  = metrics.symmetric_mae,
    symmetric_rmse = metrics.symmetric_rmse,
)


@partial(jax.jit, static_argnums=3)
def test_step(batch, state, key, net):
    imagery, mask, snake = prep(batch)
    out = {
        'imagery': imagery,
        'snake': snake,
        'mask': mask,
    }

    # sampled_pred_steps = []
    # subkeys = jax.random.split(key, 4)
    # for subkey in subkeys:
    preds, _ = net(state.params, state.buffers, key, imagery, is_training=False)

    #   sampled_pred_steps.append(pred_steps)
    if isinstance(preds, list):
        # Snake
        out['predictions'] = [preds]
        preds = preds[-1]
    elif preds.shape[:3] == imagery.shape[:3]:
        # Segmentation
        out['segmentation'] = preds
        preds = utils.snakify(preds, snake.shape[-2])
        out['predictions'] = [[preds]]
    else:
        raise ValueError("Model outputs unknown data representation")

    # Convert from normalized to to pixel coordinates
    scale = imagery.shape[1] / 2
    snake *= scale
    preds *= scale

    metrics = {}
    for m in METRICS:
        metrics[m] = jax.vmap(METRICS[m])(preds, snake)

    return metrics, out


if __name__ == '__main__':
    run = Path(sys.argv[1])
    assert run.exists()
    do_output = True

    config = yaml.load(open(run / 'config.yml'), Loader=yaml.SafeLoader)
    config['data_root'] = '../CALFIN/training/data'
    config['data_channels'] = [2]

    datasets = ['validation']# , 'validation_baumhoer', 'validation_zhang']
    loaders  = {d: get_loader(4, 1, d, config, None, subtiles=False) for d in datasets}

    # config['data_root'] = '../aicore/uc1/data/'
    # config['data_channels'] = ['SPECTRAL/BANDS/STD_2s_B8_8b']
    # loaders['TUD_test'] = get_loader(1, 4, 'test', config, drop_last=False, subtiles=False)

    for sample_batch in loaders[datasets[0]]:
        break
    img, *_ = prep(sample_batch)

    S, params, buffers = models.get_model(config, img)
    state = utils.load_state(run / 'latest.pkl')
    net = S.apply

    for dataset, loader in loaders.items():
        test_key = jax.random.PRNGKey(27)
        test_metrics = {}
        for batch in tqdm(loader, desc=dataset):
            test_key, subkey = jax.random.split(test_key)
            metrics, output = test_step(batch, state, subkey, net)

            for m in metrics:
              if m not in test_metrics: test_metrics[m] = []
              test_metrics[m].append(metrics[m])

        logging.log_metrics(test_metrics, dataset, 0, do_wandb=False)
        # print(f'{"Metric".ljust(15)}:    mean   median      min        max')
        # for m, val in metrics.items():
        #     print(f'{m.ljust(15)}: {val.mean():7.4f}  {jnp.median(val):7.4f}  '
        #           f'{jnp.min(val):7.4f} – {jnp.max(val):8.4f}')
        # print()
        # if do_output:
        #     with open('results.csv', 'a') as f:
        #         for m, val in metrics.items():
        #             print(f'{run.stem},{config["run_id"]},{dataset},{m},'
        #                   f'{val.mean()},{jnp.median(val)},{jnp.mean(val)},{jnp.min(val)},{jnp.max(val)}',
        #                     file=f)
