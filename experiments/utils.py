import os
import torch
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm


_UQ_DICT = {}
_LOSS_DICT = {}
_BOUND_DICT = {}
_CALIBRATION_DICT = {}


def register_fn(dict=None, name=None):
    def _register(func):
        def _fn(*args, **kwargs):
            return func(*args, **kwargs)

        dict[name] = _fn
        return _fn

    return _register


def register_uq(name=None):
    return register_fn(dict=_UQ_DICT, name=name)


def register_loss(name=None):
    return register_fn(dict=_LOSS_DICT, name=name)


def register_bound(name=None):
    return register_fn(dict=_BOUND_DICT, name=name)


def register_calibration(name=None):
    return register_fn(dict=_CALIBRATION_DICT, name=name)


def get_uq(name, *args, **kwargs):
    def _f(sampled, **kfargs):
        return _UQ_DICT[name](sampled, *args, **kwargs, **kfargs)

    return _f


def get_loss(name, *args, **kwargs):
    def _f(target, l, u, **kfargs):
        return _LOSS_DICT[name](target, l, u, *args, **kwargs, **kfargs)

    return _f


def get_bound(name, *args, **kwargs):
    def _f(n, delta, loss=None, **kfargs):
        return _BOUND_DICT[name](n, delta, loss, *args, **kwargs, **kfargs)

    return _f


def get_calibration(name):
    return _CALIBRATION_DICT[name]


def original_perturbed_denoised(
    dataset, config, exp_id, sigma0, shuffle=False, N=None, n=None
):
    group_id = os.path.join(config.name, str(exp_id))
    dataset_id = f"{config.data.name}_{config.data.image_size}"

    denoising_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "denoising"
    )
    denoising_dataset_dir = os.path.join(denoising_dir, dataset_id)
    denoising_sigma_dir = os.path.join(denoising_dataset_dir, str(sigma0))
    perturbed_dir = os.path.join(denoising_sigma_dir, "perturbed", dataset.op)
    denoised_dir = os.path.join(denoising_sigma_dir, group_id, "denoised", dataset.op)
    if config.model.name == "ncsnpp":
        denoised_dir = os.path.join(denoised_dir, str(N))

    perturbed_id = dataset.ids
    if shuffle:
        perturbed_id = np.random.shuffle(perturbed_id)
    if n is not None:
        perturbed_id = dataset.ids[:n]

    def _get_original(_id):
        if config.data.dataset == "CIFAR10" or config.data.dataset == "CelebA":
            if dataset.return_img_id:
                original, _ = dataset[_id]
            else:
                original = dataset[_id]
        elif config.data.dataset == "AbdomenCT1K":
            original = dataset.transform(
                np.load(
                    os.path.join(dataset.root, "images", f"{_id}.npy"),
                    allow_pickle=True,
                )
            )
        else:
            raise NotImplementedError
        return original

    def _get_perturbed(_id):
        return torch.load(os.path.join(perturbed_dir, f"{_id}.pt"))

    def _get_denoised(_id):
        if config.model.name == "ncsnpp":
            denoised = torch.load(os.path.join(denoised_dir, f"{_id}.pt"))
        elif config.model.name == "im2im_ncsnpp":
            denoised = torch.stack(
                [
                    torch.load(os.path.join(denoised_dir, f"{_id}_l.pt")),
                    torch.load(os.path.join(denoised_dir, f"{_id}.pt")),
                    torch.load(os.path.join(denoised_dir, f"{_id}_u.pt")),
                ]
            )
        return denoised

    original = torch.stack(
        Parallel(n_jobs=32)(delayed(_get_original)(_id) for _id in tqdm(perturbed_id))
    )
    perturbed = torch.stack(
        Parallel(n_jobs=32)(delayed(_get_perturbed)(_id) for _id in tqdm(perturbed_id))
    )
    denoised = torch.stack(
        Parallel(n_jobs=32)(delayed(_get_denoised)(_id) for _id in tqdm(perturbed_id))
    )

    if config.data.dataset == "CIFAR10" or config.data.dataset == "CelebA":
        mu = std = torch.tensor((0.5, 0.5, 0.5))
        denorm = lambda x: (x * std[:, None, None]) + mu[:, None, None]
        original = denorm(original)
        perturbed = denorm(perturbed)
        denoised = denorm(denoised)
    perturbed = torch.clip(perturbed, 0, 1)
    denoised = torch.clip(denoised, 0, 1)
    return perturbed_id, original, perturbed, denoised


def calibration_results(
    config,
    exp_id,
    sigma0,
    keys=None,
    no_rcps=False,
    krcps_configs=None,
    vrcps_configs=[],
):
    group_id = os.path.join(config.name, str(exp_id))
    dataset_id = f"{config.data.name}_{config.data.image_size}"

    calibration_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "calibration"
    )
    calibration_dataset_dir = os.path.join(calibration_dir, dataset_id)
    calibration_sigma_dir = os.path.join(calibration_dataset_dir, str(sigma0))
    calibration_group_dir = os.path.join(calibration_sigma_dir, group_id)

    calibration = list(os.walk(calibration_group_dir))[0][1]
    results = {}
    for _calibration in tqdm(calibration):
        if _calibration == "rcps" and no_rcps:
            continue
        if "k_rcps" in _calibration and krcps_configs is not None:
            if _calibration not in krcps_configs:
                continue
        if "v_rcps" in _calibration and vrcps_configs is not None:
            if _calibration not in vrcps_configs:
                continue

        _calibration_dir = os.path.join(calibration_group_dir, _calibration)

        if keys is None:
            keys = ["val_idx", "lambda", "val_loss", "val_l", "val_u", "val_i"]
        results[_calibration] = {
            key: torch.load(os.path.join(_calibration_dir, f"{key}.pt")) for key in keys
        }
    return results
