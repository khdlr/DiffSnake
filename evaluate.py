from functools import partial
import jax
import jax.numpy as jnp

from lib import losses
from lib.utils import prep
from lib.diffusion import ddim_sample
from einops import repeat

from jax.experimental.host_callback import id_print


METRICS = dict(
    mae=losses.mae,
    rmse=losses.rmse,
    symmetric_mae=losses.symmetric_mae,
    symmetric_rmse=losses.symmetric_rmse,
)


@partial(jax.jit, static_argnums=3)
def test_step(batch, state, key, model):
    imagery, mask, contour = prep(batch)
    B = imagery.shape[0]
    S = 4

    original_contour = contour
    contour = repeat(contour, 'B T C -> B S T C', S=S)
    ts = repeat(jnp.linspace(0, 1000, 50), 'T -> T B', B=B)
    init = jax.random.normal(key, contour.shape)

    img_features = model.get_features(state.params, imagery)
    # sample_single = partial(ddim_sample, model, state.params, img_features, ts)
    # trajectories = jax.vmap(sample_single, in_axes=1, out_axes=1)(init)
    trajectories = ddim_sample(model, state.params, img_features, ts, init)

    terms = {
        "imagery": imagery,
        "contour": contour,
        "mask": mask,
        'snake_steps': trajectories,
        'snake': trajectories[:, :, -1],
    }

    # Convert from normalized to to pixel coordinates
    scale = imagery.shape[1] / 2
    for key in ["snake", "snake_steps", "contour"]:
        terms[key] = jax.tree_map(lambda x: scale * (1.0 + x), terms[key])

    # TODO: uncertainty measures
    metrics = {}
    for m in METRICS:
        metrics[m] = jnp.mean(jax.vmap(partial(losses.call_loss, METRICS[m]))(terms)[0])

    terms['contour'] = jax.tree_map(lambda x: scale * (1.0 + x), original_contour)
    return metrics, terms


if __name__ == "__main__":
    run = Path(sys.argv[1])
    assert run.exists()
    do_output = True

    config = yaml.load(open(run / "config.yml"), Loader=yaml.SafeLoader)
    if "dataset" in config and config["dataset"] == "TUD-MS":
        # datasets = ['TEST' , '', 'validation_zhang']
        loaders = {"TUD-MS": get_loader(4, 1, "test", config, None, subtiles=False)}
    else:
        config["dataset"] = "CALFIN"
        config["data_root"] = "../CALFIN/training/data"
        config["data_channels"] = [2]

        datasets = ["validation", "validation_baumhoer", "validation_zhang"]
        loaders = {
            d: get_loader(4, 1, d, config, None, subtiles=False) for d in datasets
        }

        config["dataset"] = "TUD"
        config["data_root"] = "../aicore/uc1/data/"
        config["data_channels"] = ["SPECTRAL/BANDS/STD_2s_B8_8b"]
        loaders["TUD_test"] = get_loader(4, 1, "test", config, subtiles=False)

    for sample_batch in list(loaders.values())[0]:
        img, *_ = prep(sample_batch)
        break

    S, params, buffers = models.get_model(config, img)
    state = utils.load_state(run / "latest.pkl")
    net = S.apply

    img_root = run / "imgs"
    img_root.mkdir(exist_ok=True)

    all_metrics = {}
    for dataset, loader in loaders.items():
        test_key = jax.random.PRNGKey(27)
        test_metrics = {}

        img_dir = img_root / dataset
        img_dir.mkdir(exist_ok=True)
        dsidx = 0
        for batch in tqdm(loader, desc=dataset):
            test_key, subkey = jax.random.split(test_key)
            metrics, output = test_step(batch, state, subkey, net)

            for m in metrics:
                if m not in test_metrics:
                    test_metrics[m] = []
                test_metrics[m].append(metrics[m])

            for i in range(len(output["imagery"])):
                o = jax.tree_map(lambda x: x[i], output)
                raw = Image.fromarray(
                    (255 * np.asarray(o["imagery"][..., 0])).astype(np.uint8)
                )
                raw_path = Path(f"base_imgs/{dataset}/{dsidx:03d}.jpg")
                raw_path.parent.mkdir(exist_ok=True, parents=True)
                raw.save(f"base_imgs/{dataset}/{dsidx:03d}.jpg")
                base = 0.5 * (o["imagery"] + 1.0)
                logging.draw_image(
                    base, o["contour"], o["snake"], img_dir / f"{dsidx:03d}.pdf"
                )
                logging.draw_steps(
                    base,
                    o["contour"],
                    o["snake_steps"],
                    img_dir / f"{dsidx:03d}_steps.pdf",
                )
                dsidx += 1

        logging.log_metrics(test_metrics, dataset, 0, do_wandb=False)
        for m in test_metrics:
            all_metrics[f"{dataset}/{m}"] = np.mean(test_metrics[m])

    with (run / "new_metrics.json").open("w") as f:
        print(all_metrics, file=f)
