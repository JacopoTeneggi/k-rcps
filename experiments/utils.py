import os
import torch
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm


def denoising_results(dataset, config, shuffle=False, n=None):
    denoising_dir = os.path.join(os.path.dirname(__file__), "denoising")
    denoising_dataset_dir = os.path.join(denoising_dir, config.data.name)
    perturbed_dir = os.path.join(denoising_dataset_dir, "perturbed")
    denoised_dir = os.path.join(denoising_dataset_dir, config.name)

    perturbed_id = dataset.ids
    if shuffle:
        perturbed_id = np.random.shuffle(perturbed_id)
    if n is not None:
        perturbed_id = dataset.ids[:n]

    def _get_original(_id):
        if config.data.dataset == "CelebA":
            if dataset.return_img_id:
                original, _ = dataset[_id]
            else:
                original = dataset[_id]
        if config.data.dataset == "AbdomenCT-1K":
            original = dataset.transform(
                np.load(
                    os.path.join(dataset.root, "images", f"{_id}.npy"),
                    allow_pickle=True,
                )
            )
        return original

    def _get_perturbed(_id):
        return torch.load(os.path.join(perturbed_dir, f"{_id}.pt"))

    def _get_denoised(_id):
        if config.model.name in ["ncsnpp", "mc_dropout"]:
            denoised = torch.load(os.path.join(denoised_dir, f"{_id}.pt"))
        if config.model.name in ["im2im_ncsnpp", "ncsnpp_conffusion"]:
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

    if config.data.dataset == "CelebA":
        mu = std = torch.tensor((0.5, 0.5, 0.5))
        denorm = lambda x: (x * std[:, None, None]) + mu[:, None, None]
        original = denorm(original)
        perturbed = denorm(perturbed)
        denoised = denorm(denoised)
    perturbed = torch.clamp(perturbed, 0, 1)
    denoised = torch.clamp(denoised, 0, 1)
    return perturbed_id, original, perturbed, denoised
